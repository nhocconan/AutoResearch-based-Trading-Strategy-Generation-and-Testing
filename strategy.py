#!/usr/bin/env python3
"""
Experiment #696: 12h Primary + 1d HTF — Donchian Breakout + HMA Trend + RSI Filter

Hypothesis: Donchian breakouts capture major moves while HMA trend filter prevents
counter-trend whipsaws. Simpler than #692 (no BB/Keltner squeeze) to ensure trade
frequency on ALL symbols. RSI filter adds timing precision without over-constraining.

Key Changes from #692:
1. Donchian(20) breakout instead of BB/Keltner squeeze (more reliable, more trades)
2. Single HTF filter (1d HMA only, not 1d+1w) to reduce over-filtering
3. Looser RSI thresholds (35/65 not 30/70) to ensure entries on BTC/ETH
4. Simpler hold logic — maintain position until trend reverses or stoploss
5. Remove ADX filter (was preventing trades in #692)

Why this should work:
- Donchian breakouts proven in #686 (SOL Sharpe +0.782)
- 12h TF worked in #682 (Sharpe=0.404) with simpler logic
- Single HTF filter reduces over-constraint that caused 0 trades in #693/#695
- ATR trailing stop protects against 2022-style crashes

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_rsi_atr_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_donchian(high, low, period=20):
    """Donchian Channels - highest high and lowest low over period."""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2
    return upper, lower, mid

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
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i-1])
        tr3 = np.abs(low[i] - close[i-1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (12h) indicators
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    rsi_12h = calculate_rsi(close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    sma_50 = calculate_sma(close, period=50)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
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
    
    for i in range(250, n):  # Need 200 for SMA + buffer for HTF alignment
        # Skip if indicators not ready
        if np.isnan(rsi_12h[i]) or np.isnan(atr_12h[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            continue
        if atr_12h[i] <= 1e-10:
            continue
        
        # === TREND BIAS (1d HMA) ===
        trend_bullish = close[i] > hma_1d_aligned[i]
        trend_bearish = close[i] < hma_1d_aligned[i]
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        below_sma50 = close[i] < sma_50[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI FILTER ===
        rsi_neutral = 35 <= rsi_12h[i] <= 65
        rsi_bullish = rsi_12h[i] > 45
        rsi_bearish = rsi_12h[i] < 55
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # Reduce size in high volatility
        atr_median = np.nanmedian(atr_12h[max(0,i-100):i+1])
        atr_ratio = atr_12h[i] / (atr_median + 1e-10)
        if atr_ratio > 1.8:
            current_size = REDUCED_SIZE
        
        # === LONG ENTRY: Donchian Breakout + Trend Confirmation ===
        # Primary: Breakout above Donchian upper + bullish 1d HMA + above SMA50
        if close[i] > donchian_upper[i-1] and trend_bullish and above_sma50:
            # Strong signal if also above SMA200
            if above_sma200 and rsi_bullish:
                desired_signal = current_size
            # Weaker signal but still valid
            elif rsi_12h[i] > 40:
                desired_signal = current_size * 0.7
        
        # === SHORT ENTRY: Donchian Breakdown + Trend Confirmation ===
        # Primary: Breakdown below Donchian lower + bearish 1d HMA + below SMA50
        elif close[i] < donchian_lower[i-1] and trend_bearish and below_sma50:
            # Strong signal if also below SMA200
            if below_sma200 and rsi_bearish:
                desired_signal = -current_size
            # Weaker signal but still valid
            elif rsi_12h[i] < 60:
                desired_signal = -current_size * 0.7
        
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
                # Hold long if 1d HMA still bullish and RSI not extreme
                if trend_bullish and rsi_12h[i] < 80:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 1d HMA still bearish and RSI not extreme
                if trend_bearish and rsi_12h[i] > 20:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        # Long exit: RSI overbought OR trend reverses below 1d HMA
        if in_position and position_side > 0:
            if rsi_12h[i] > 85:
                desired_signal = 0.0
            elif close[i] < hma_1d_aligned[i]:
                desired_signal = 0.0
        
        # Short exit: RSI oversold OR trend reverses above 1d HMA
        if in_position and position_side < 0:
            if rsi_12h[i] < 15:
                desired_signal = 0.0
            elif close[i] > hma_1d_aligned[i]:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.9:
                desired_signal = BASE_SIZE
            elif desired_signal >= REDUCED_SIZE * 0.9:
                desired_signal = REDUCED_SIZE
            else:
                desired_signal = 0.0
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.9:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -REDUCED_SIZE * 0.9:
                desired_signal = -REDUCED_SIZE
            else:
                desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
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