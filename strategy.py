#!/usr/bin/env python3
"""
Experiment #018: 30m Primary + 4h/1d HTF — Fisher Transform + BB Mean Reversion

Hypothesis: After 17 experiments, the key insight is that bear/range markets (2025)
require mean reversion strategies, NOT trend following. Fisher Transform excels at
catching reversals at extremes, especially when combined with Bollinger Band extremes
and HTF trend filter.

Key innovations:
1. Fisher Transform (period=9) - normalized price oscillator, extreme values (-2/+2) signal reversals
2. Bollinger Band %B - only enter when price at BB extremes (<0.05 or >0.95)
3. 4h HMA trend filter - only long when 4h bullish, only short when 4h bearish
4. 1d HMA major bias - avoid counter-trend trades against daily
5. Volume filter - only trade when volume > 0.7x 20-bar average
6. ATR(7)/ATR(30) vol spike filter - enter when vol elevated (ratio > 1.5)

Entry Logic (LONG):
- 4h close > 4h HMA(21) + 1d close > 1d HMA(21) (HTF bullish)
- Fisher < -1.5 (oversold reversal signal)
- BB %B < 0.10 (price at lower band extreme)
- Volume > 0.7x avg(20)
- ATR ratio > 1.3 (elevated volatility = mean reversion opportunity)

Entry Logic (SHORT):
- 4h close < 4h HMA(21) + 1d close < 1d HMA(21) (HTF bearish)
- Fisher > +1.5 (overbought reversal signal)
- BB %B > 0.90 (price at upper band extreme)
- Volume > 0.7x avg(20)
- ATR ratio > 1.3

Risk: 2.5x ATR trailing stop, signal→0 when stopped out
Size: 0.25 (smaller for 30m TF to reduce fee drag)
Target: Sharpe>0.3, trades>30/symbol train, >3/symbol test, DD>-40%, <80 trades/year
Timeframe: 30m (use 4h/1d for direction, 30m for entry timing)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_bb_volregime_v1"
timeframe = "30m"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Extreme values (-2 to +2) indicate reversal points
    """
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    for i in range(period, n):
        # Calculate typical price range
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        range_val = highest - lowest
        if range_val < 1e-10:
            fisher[i] = 0.0
            fisher_signal[i] = fisher[i-1] if i > 0 else 0.0
            continue
        
        # Normalize price to 0-1 range
        normalized = (hl2 - lowest) / range_val
        
        # Constrain to 0.001-0.999 to avoid log(0)
        normalized = np.clip(normalized, 0.001, 0.999)
        
        # Fisher calculation
        fisher_val = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Smooth with previous value
        if i > period and not np.isnan(fisher[i-1]):
            fisher[i] = 0.67 * fisher_val + 0.33 * fisher[i-1]
        else:
            fisher[i] = fisher_val
        
        # Signal line (previous fisher)
        fisher_signal[i] = fisher[i-1] if i > 0 else fisher[i]
    
    return fisher, fisher_signal

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands with %B indicator"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    # Rolling mean and std
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    # %B indicator: where price is within bands (0=lower, 1=upper)
    pct_b = np.full(n, np.nan)
    for i in range(period, n):
        band_width = upper[i] - lower[i]
        if band_width < 1e-10:
            pct_b[i] = 0.5
        else:
            pct_b[i] = (close[i] - lower[i]) / band_width
    
    return upper, lower, pct_b

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Hull Moving Average - smoother, more responsive than EMA"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA calculations
    wma_half = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma_full = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Raw HMA
    raw_hma = 2.0 * wma_half - wma_full
    
    # Smooth HMA
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (30m) indicators
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    bb_upper, bb_lower, bb_pct_b = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    
    # Volume average (20 bars)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    SIZE = 0.25  # Smaller size for 30m TF to reduce fee drag
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(fisher[i]) or np.isnan(bb_pct_b[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND FILTER (4h + 1d HMA) ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === VOLATILITY REGIME FILTER ===
        atr_ratio = atr_7[i] / atr_30[i] if atr_30[i] > 1e-10 else 0.0
        vol_elevated = atr_ratio > 1.3  # Volatility spike = mean reversion opportunity
        
        # === VOLUME FILTER ===
        vol_ok = volume[i] > 0.7 * vol_avg[i]
        
        # === FISHER TRANSFORM REVERSAL SIGNAL ===
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # === BOLLINGER BAND EXTREME ===
        bb_extreme_low = bb_pct_b[i] < 0.10
        bb_extreme_high = bb_pct_b[i] > 0.90
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # Long entry: HTF bullish + Fisher oversold + BB low + vol filter
        if hma_4h_bull and hma_1d_bull and fisher_oversold and bb_extreme_low and vol_ok and vol_elevated:
            desired_signal = SIZE
        
        # Short entry: HTF bearish + Fisher overbought + BB high + vol filter
        elif hma_4h_bear and hma_1d_bear and fisher_overbought and bb_extreme_high and vol_ok and vol_elevated:
            desired_signal = -SIZE
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
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
        
        signals[i] = final_signal
    
    return signals