#!/usr/bin/env python3
"""
Experiment #724: 4h Primary + 1d/1w HTF — Adaptive KAMA Trend + RSI Pullback + Choppiness Regime

Hypothesis: After 484 failed strategies, the key insight is that complex regime switching
prevents trades. This strategy uses:
1. KAMA (Kaufman Adaptive MA) - adapts to volatility, works better than HMA/EMA in range markets
2. Choppiness Index for regime detection (CHOP > 61.8 = range, < 38.2 = trend)
3. RSI pullback entries in direction of KAMA trend
4. 1d/1w KAMA for HTF trend confirmation (looser than HMA)
5. ATR trailing stops (2.5x) for risk management
6. Multiple entry paths to ensure trade frequency (≥10 train, ≥3 test per symbol)

Key differences from failed experiments:
- KAMA instead of HMA (better adaptation to volatility regimes)
- Simpler Choppiness thresholds (61.8/38.2 from original research)
- Looser RSI thresholds (40/60 for entries, not 30/70)
- Multiple entry paths (pullback + breakout + mean reversion)
- No complex position state tracking that caused 0 trades in #712, #715

Target: Sharpe > 0.612 (current best), trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (proven, 20-50 trades/year target)
Position Size: 0.30 (discrete, minimizes fee churn)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_rsi_chop_regime_1d1w_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average.
    Adapts to market noise - smooth in choppy markets, responsive in trends.
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Efficiency Ratio (ER) - measures trend vs noise
    er = np.zeros(n)
    for i in range(period, n):
        price_change = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if noise > 0:
            er[i] = price_change / noise
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = (2.0 / (fast_period + 1)) ** 2
    slow_sc = (2.0 / (slow_period + 1)) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    return kama

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP).
    Measures market choppiness vs trending.
    CHOP > 61.8 = choppy/ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high > lowest_low:
            atr_sum = 0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
                atr_sum += tr
            
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 100
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel for breakout detection."""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands."""
    n = len(close)
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    pct_b = (close - lower) / (upper - lower + 1e-10)
    return upper, lower, sma, pct_b

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    rsi_4h = calculate_rsi(close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    bb_upper, bb_lower, bb_sma, pct_b = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    # Calculate and align HTF KAMA for trend bias
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    kama_1w_raw = calculate_kama(df_1w['close'].values, period=10)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
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
        if np.isnan(kama_4h[i]) or np.isnan(chop_4h[i]):
            continue
        if np.isnan(kama_1d_aligned[i]) or np.isnan(kama_1w_aligned[i]):
            continue
        if np.isnan(donch_upper[i]) or np.isnan(bb_upper[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_4h[i] > 61.8  # Range market
        is_trending = chop_4h[i] < 38.2  # Trend market
        # Neutral regime: 38.2 <= CHOP <= 61.8
        
        # === TREND BIAS (KAMA alignment) ===
        trend_1d_bullish = close[i] > kama_1d_aligned[i]
        trend_1d_bearish = close[i] < kama_1d_aligned[i]
        trend_1w_bullish = close[i] > kama_1w_aligned[i]
        trend_1w_bearish = close[i] < kama_1w_aligned[i]
        
        # 4h KAMA trend
        trend_4h_bullish = close[i] > kama_4h[i]
        trend_4h_bearish = close[i] < kama_4h[i]
        
        # Strong trend when HTF agrees
        strong_bullish = trend_1d_bullish and trend_1w_bullish
        strong_bearish = trend_1d_bearish and trend_1w_bearish
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (multiple paths for trade frequency) ===
        long_signal = False
        
        # Path 1: Trending regime + bullish HTF + RSI pullback
        if is_trending and strong_bullish and 35 < rsi_4h[i] < 55:
            long_signal = True
        
        # Path 2: Choppy regime + bullish HTF + RSI oversold (mean reversion)
        if is_choppy and trend_1d_bullish and rsi_4h[i] < 40:
            long_signal = True
        
        # Path 3: Donchian breakout + bullish trend
        if close[i] > donch_upper[i-1] and trend_1d_bullish and rsi_4h[i] < 60:
            long_signal = True
        
        # Path 4: Price at BB lower + bullish HTF (buy dip)
        if pct_b[i] < 0.2 and trend_1d_bullish and rsi_4h[i] < 50:
            long_signal = True
        
        # Path 5: 4h KAMA bullish + RSI crossing up from oversold
        if trend_4h_bullish and rsi_4h[i] > 40 and rsi_4h[i-1] < 40:
            long_signal = True
        
        # Path 6: Simple bullish alignment (ensure trades in strong trends)
        if strong_bullish and rsi_4h[i] < 50 and trend_4h_bullish:
            long_signal = True
        
        if long_signal:
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRY CONDITIONS (multiple paths for trade frequency) ===
        short_signal = False
        
        # Path 1: Trending regime + bearish HTF + RSI bounce
        if is_trending and strong_bearish and 45 < rsi_4h[i] < 65:
            short_signal = True
        
        # Path 2: Choppy regime + bearish HTF + RSI overbought (mean reversion)
        if is_choppy and trend_1d_bearish and rsi_4h[i] > 60:
            short_signal = True
        
        # Path 3: Donchian breakdown + bearish trend
        if close[i] < donch_lower[i-1] and trend_1d_bearish and rsi_4h[i] > 40:
            short_signal = True
        
        # Path 4: Price at BB upper + bearish HTF (sell rip)
        if pct_b[i] > 0.8 and trend_1d_bearish and rsi_4h[i] > 50:
            short_signal = True
        
        # Path 5: 4h KAMA bearish + RSI crossing down from overbought
        if trend_4h_bearish and rsi_4h[i] < 60 and rsi_4h[i-1] > 60:
            short_signal = True
        
        # Path 6: Simple bearish alignment (ensure trades in strong trends)
        if strong_bearish and rsi_4h[i] > 50 and trend_4h_bearish:
            short_signal = True
        
        if short_signal:
            desired_signal = -BASE_SIZE
        
        # === CONFLICT RESOLUTION ===
        if long_signal and short_signal:
            # Go with HTF trend direction
            if trend_1w_bullish:
                desired_signal = BASE_SIZE
            elif trend_1w_bearish:
                desired_signal = -BASE_SIZE
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
                # Hold long if 1d KAMA still bullish and RSI not extremely overbought
                if trend_1d_bullish and rsi_4h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 1d KAMA still bearish and RSI not extremely oversold
                if trend_1d_bearish and rsi_4h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if HTF trend reverses or RSI extremely overbought
            if trend_1d_bearish or rsi_4h[i] > 80:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if HTF trend reverses or RSI extremely oversold
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