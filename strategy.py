#!/usr/bin/env python3
"""
Experiment #721: 4h Primary + 1d/1w HTF — Volatility Breakout with Trend Filter

Hypothesis: After analyzing 482 failed strategies, the key issue is overly complex
regime detection that prevents trades. This strategy uses:
1. 1d HMA for strong trend bias (proven in current best Sharpe=0.612)
2. 4h Bollinger Band squeeze detection for volatility breakouts
3. 4h RSI for entry timing (looser thresholds to ensure trade frequency)
4. 1w HMA for ultra-long-term trend confirmation
5. Simple ATR stoploss (2.5x) without complex position tracking

Key differences from failed #709:
- Removed complex regime switching (ADX thresholds caused 0 trades)
- Looser RSI thresholds (35/65 instead of 30/70)
- BB squeeze detection ensures we enter on volatility expansion
- Simpler hold/exit logic (no complex position state tracking)
- Ensure signals generate trades by having multiple entry paths

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (proven to work, 20-50 trades/year target)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_vol_breakout_hma_rsi_1d1w_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands with squeeze detection."""
    n = len(close)
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    pct_b = (close - lower) / (upper - lower + 1e-10)
    
    # Bandwidth for squeeze detection
    bandwidth = (upper - lower) / (sma + 1e-10)
    
    return upper, lower, sma, pct_b, bandwidth

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_donchian(high, low, period=20):
    """Donchian Channel for breakout detection."""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (4h) indicators
    rsi_4h = calculate_rsi(close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_sma, pct_b, bb_bandwidth = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate BB bandwidth percentile for squeeze detection
    bb_bw_percentile = pd.Series(bb_bandwidth).rolling(window=100, min_periods=100).apply(
        lambda x: np.percentile(x[~np.isnan(x)], 20) if len(x[~np.isnan(x)]) > 0 else np.nan
    ).values
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):  # Need buffer for all indicators + HTF alignment
        # Skip if indicators not ready
        if np.isnan(rsi_4h[i]) or np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(bb_upper[i]) or np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(sma_200[i]) or np.isnan(donch_upper[i]) or np.isnan(bb_bandwidth[i]):
            continue
        
        # === TREND BIAS (1d and 1w HTF HMA) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        trend_1w_bullish = close[i] > hma_1w_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # Strong trend when both 1d and 1w agree
        strong_bullish = trend_1d_bullish and trend_1w_bullish
        strong_bearish = trend_1d_bearish and trend_1w_bearish
        
        # === VOLATILITY SQUEEZE DETECTION ===
        # Low bandwidth = squeeze, expect breakout soon
        bb_squeeze = bb_bandwidth[i] < np.nanpercentile(bb_bandwidth[:i+1], 25) if i > 100 else False
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === LONG ENTRY CONDITIONS (multiple paths to ensure trades) ===
        long_signal = False
        
        # Path 1: Strong bullish trend + RSI pullback
        if strong_bullish and rsi_4h[i] < 45 and above_sma200:
            long_signal = True
        
        # Path 2: BB squeeze + bullish trend + RSI not overbought
        if bb_squeeze and trend_1d_bullish and rsi_4h[i] < 55:
            long_signal = True
        
        # Path 3: Donchian breakout + bullish trend
        if close[i] > donch_upper[i-1] and trend_1d_bullish and rsi_4h[i] < 60:
            long_signal = True
        
        # Path 4: RSI deeply oversold + above 1d HMA (mean reversion in uptrend)
        if rsi_4h[i] < 35 and trend_1d_bullish:
            long_signal = True
        
        # Path 5: Price near BB lower + bullish trend (buy the dip)
        if pct_b[i] < 0.15 and trend_1d_bullish and rsi_4h[i] < 50:
            long_signal = True
        
        if long_signal:
            desired_signal = current_size
        
        # === SHORT ENTRY CONDITIONS (multiple paths to ensure trades) ===
        short_signal = False
        
        # Path 1: Strong bearish trend + RSI bounce
        if strong_bearish and rsi_4h[i] > 55 and below_sma200:
            short_signal = True
        
        # Path 2: BB squeeze + bearish trend + RSI not oversold
        if bb_squeeze and trend_1d_bearish and rsi_4h[i] > 45:
            short_signal = True
        
        # Path 3: Donchian breakdown + bearish trend
        if close[i] < donch_lower[i-1] and trend_1d_bearish and rsi_4h[i] > 40:
            short_signal = True
        
        # Path 4: RSI deeply overbought + below 1d HMA (mean reversion in downtrend)
        if rsi_4h[i] > 65 and trend_1d_bearish:
            short_signal = True
        
        # Path 5: Price near BB upper + bearish trend (sell the rip)
        if pct_b[i] > 0.85 and trend_1d_bearish and rsi_4h[i] > 50:
            short_signal = True
        
        if short_signal:
            desired_signal = -current_size
        
        # === CONFLICT RESOLUTION ===
        # If both long and short signals, go with stronger trend (1w HMA)
        if long_signal and short_signal:
            if trend_1w_bullish:
                desired_signal = current_size
            elif trend_1w_bearish:
                desired_signal = -current_size
            else:
                desired_signal = 0.0
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 1d HMA still bullish and RSI not extremely overbought
                if trend_1d_bullish and rsi_4h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 1d HMA still bearish and RSI not extremely oversold
                if trend_1d_bearish and rsi_4h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses or RSI extremely overbought
            if trend_1d_bearish or rsi_4h[i] > 80:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses or RSI extremely oversold
            if trend_1d_bullish or rsi_4h[i] < 20:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals