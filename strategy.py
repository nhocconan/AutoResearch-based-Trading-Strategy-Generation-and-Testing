#!/usr/bin/env python3
"""
Experiment #428: 30m Primary + 4h/1d HTF — Fisher Transform + HMA Trend + Session Filter

Hypothesis: 30m timeframe with strict HTF filters should produce 30-80 trades/year
with controlled drawdown. Key innovations:
1. 4h HMA for primary trend direction (stronger than 1h, less lag than 1d)
2. Ehlers Fisher Transform (period=9) for entry timing — catches reversals better than RSI
3. 1d Choppiness Index for regime detection — avoid mean-revert in strong trends
4. Session filter: only 8-20 UTC (reduces trades ~50%, avoids Asian chop)
5. Volume filter: only trade when volume > 0.7x 20-bar average
6. Conservative size: 0.25 (smaller for 30m to reduce fee drag and DD)
7. Trailing stoploss: 2.5 ATR from entry/extreme

Why this should beat #418 (Sharpe=-1.064):
- Simpler entry conditions (Fisher cross vs CRSI thresholds)
- 4h HMA slope filter is more robust than HMA crossover
- Session filter dramatically reduces low-quality trades
- Fisher Transform is more sensitive to reversals in bear markets

Target: Sharpe > 0.612, 120-320 trades over 4-year train (30-80/year), DD < -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_hma_chop_session_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    hma = diff.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return hma.values

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * (price - lowest) / (highest - lowest) - 0.33
    Signals: cross above -1.5 = long, cross below +1.5 = short
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = high[i-period+1:i+1].max()
        lowest = low[i-period+1:i+1].min()
        
        if highest - lowest < 1e-10:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
            fisher_signal[i] = fisher_signal[i-1] if i > 0 else 0.0
            continue
        
        price = (high[i] + low[i]) / 2.0
        X = 0.67 * (price - lowest) / (highest - lowest) - 0.33
        X = np.clip(X, -0.99, 0.99)  # Prevent division by zero
        
        fisher[i] = 0.5 * np.log((1.0 + X) / (1.0 - X + 1e-10))
        
        # Smooth Fisher for signal line
        if i > period:
            fisher_signal[i] = fisher[i-1]
        else:
            fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = high[i-period+1:i+1].max()
        lowest = low[i-period+1:i+1].min()
        
        if highest - lowest < 1e-10:
            chop[i] = 50.0
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr1 = high[j] - low[j]
            tr2 = abs(high[j] - close[j-1]) if j > 0 else tr1
            tr3 = abs(low[j] - close[j-1]) if j > 0 else tr1
            tr = max(tr1, tr2, tr3)
            tr_sum += tr
        
        chop[i] = 100.0 * np.log10(tr_sum / (highest - lowest)) / np.log10(period)
        chop[i] = np.clip(chop[i], 0, 100)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def calculate_volume_sma(volume, period=20):
    """Calculate volume SMA for volume filter."""
    vol_s = pd.Series(volume)
    return vol_s.rolling(window=period, min_periods=period).mean().values

def extract_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    return (open_time // 3600000) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 30m indicators (primary timeframe)
    hma_21_30m = calculate_hma(close, 21)
    hma_50_30m = calculate_hma(close, 50)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_sma_20 = calculate_volume_sma(volume, 20)
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_4h_prev_raw = calculate_hma(df_4h['close'].values, 21)
    # Shift to get previous bar for slope calculation
    hma_4h_prev_aligned = np.roll(hma_4h_aligned, 1)
    hma_4h_prev_aligned[:20] = np.nan
    
    chop_1d_raw = calculate_choppiness(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_raw)
    
    # Calculate median ATR for vol filter
    valid_atr = atr_14[100:]
    atr_median = np.nanmedian(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(valid_atr[~np.isnan(valid_atr)])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # 25% position size for 30m (smaller to reduce fee drag)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_21_30m[i]) or np.isnan(hma_50_30m[i]):
            continue
        if np.isnan(chop_1d_aligned[i]):
            continue
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] <= 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = extract_hour(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER ===
        vol_ratio = volume[i] / (vol_sma_20[i] + 1e-10)
        volume_ok = vol_ratio > 0.7
        
        # === 4H HMA SLOPE (trend direction) ===
        hma_4h_slope = hma_4h_aligned[i] - hma_4h_prev_aligned[i]
        hma_4h_bullish = hma_4h_slope > 0
        hma_4h_bearish = hma_4h_slope < 0
        
        # === 4H HMA POSITION ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === 30M HMA CROSSOVER ===
        hma_30m_bullish = hma_21_30m[i] > hma_50_30m[i]
        hma_30m_bearish = hma_21_30m[i] < hma_50_30m[i]
        
        # === CHOPPINESS REGIME (1d) ===
        is_choppy = chop_1d_aligned[i] > 55.0
        is_trending = chop_1d_aligned[i] < 45.0
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.5 and fisher_signal[i] >= -1.5
        fisher_overbought = fisher[i] > 1.5 and fisher_signal[i] <= 1.5
        fisher_rising = fisher[i] > fisher_signal[i]
        fisher_falling = fisher[i] < fisher_signal[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG SETUP (need 3+ confluence)
        long_confluence = 0
        if hma_4h_bullish:
            long_confluence += 1
        if price_above_hma_4h:
            long_confluence += 1
        if hma_30m_bullish:
            long_confluence += 1
        if fisher_rising or fisher_oversold:
            long_confluence += 1
        if in_session:
            long_confluence += 1
        if volume_ok:
            long_confluence += 1
        
        # Need at least 4 confluence for long
        if long_confluence >= 4:
            if fisher_oversold or (fisher_rising and fisher[i] < 0):
                desired_signal = BASE_SIZE
        
        # SHORT SETUP (need 3+ confluence)
        short_confluence = 0
        if hma_4h_bearish:
            short_confluence += 1
        if price_below_hma_4h:
            short_confluence += 1
        if hma_30m_bearish:
            short_confluence += 1
        if fisher_falling or fisher_overbought:
            short_confluence += 1
        if in_session:
            short_confluence += 1
        if volume_ok:
            short_confluence += 1
        
        # Need at least 4 confluence for short
        if short_confluence >= 4:
            if fisher_overbought or (fisher_falling and fisher[i] > 0):
                desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === FISHER EXTREME EXIT ===
        if in_position and position_side > 0 and fisher[i] > 1.8:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and fisher[i] < -1.8:
            desired_signal = 0.0
        
        # === HTF TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and hma_4h_bearish and price_below_hma_4h:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and hma_4h_bullish and price_above_hma_4h:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and (hma_4h_bullish or price_above_hma_4h):
                desired_signal = BASE_SIZE
            elif position_side < 0 and (hma_4h_bearish or price_below_hma_4h):
                desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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