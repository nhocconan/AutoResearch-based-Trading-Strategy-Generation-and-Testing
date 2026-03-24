#!/usr/bin/env python3
"""
Experiment #200: 6h Primary + 1d/1w HTF — Fisher Transform + ADX Regime + Volume Confirm

Hypothesis: 6h timeframe sits in a sweet spot between 4h (too noisy) and 12h (too slow).
Using Fisher Transform for entry timing (proven in bear/range markets) + ADX regime filter
+ Weekly trend bias should capture reversals while avoiding whipsaws.

Key Components:
1. 1w HMA(50): Major trend bias (very slow, stable through 2022 crash)
2. 1d ADX(14): Regime detection (ADX>25=trend, ADX<20=range)
3. 6h Fisher Transform(9): Entry timing (crosses above -1.5=long, below +1.5=short)
4. Volume ratio: Current volume / 20-bar avg > 1.2 for confirmation

Why Fisher Transform:
- Normalizes price to Gaussian distribution
- Extremes (-2 to +2) are statistically significant
- Works well in bear markets (2025 test period is bearish/range)
- Catches reversals better than RSI in choppy conditions

Position sizing: 0.25 base, 0.30 for strong confluence (all 3 HTF align)
Stoploss: 2.5x ATR trailing
Target: 30-60 trades/year, Sharpe>0.40 (beat current 6h best)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_adx_volume_1w1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

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

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index - measures trend strength
    ADX > 25 = trending market
    ADX < 20 = ranging/choppy market
    """
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smooth with Wilder's method (EMA with alpha = 1/period)
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    dx[:] = np.nan
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Formula: Fisher = 0.5 * ln((1 + X) / (1 - X))
    Where X = 0.66 * ((price - LL) / (HH - LL) - 0.5) + 0.66 * prev_X
    
    Entry signals:
    - Long: Fisher crosses above -1.5 (extreme oversold reversal)
    - Short: Fisher crosses below +1.5 (extreme overbought reversal)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_prev = np.zeros(n)
    fisher_prev[:] = np.nan
    
    # Calculate median price
    median = (high + low) / 2.0
    
    x_value = 0.0
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            # Normalize price to 0-1 range
            normalized = (median[i] - lowest_low) / price_range
            
            # Calculate X with smoothing
            x_value = 0.66 * ((normalized - 0.5) + 0.66 * x_value)
            
            # Clamp X to avoid log domain errors
            x_value = max(-0.999, min(0.999, x_value))
            
            # Fisher Transform
            fisher[i] = 0.5 * np.log((1.0 + x_value) / (1.0 - x_value))
            if i > period:
                fisher_prev[i] = fisher[i-1]
    
    return fisher, fisher_prev

def calculate_volume_ratio(volume, period=20):
    """Current volume / rolling average volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    
    # Handle division by zero
    vol_ratio[vol_avg < 1e-10] = np.nan
    
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate and align 1d ADX for regime detection
    adx_1d_raw = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_raw)
    
    # Calculate primary (6h) indicators
    atr = calculate_atr(high, low, close, period=14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25  # 25% base position size
    SIZE_STRONG = 0.30  # 30% for strong signals
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Start after all indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(adx_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (1d ADX) ===
        adx_value = adx_1d_aligned[i]
        is_trending = adx_value > 25.0
        is_ranging = adx_value < 20.0
        # 20-25 is transition zone
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_long = fisher_prev[i] < -1.5 and fisher[i] >= -1.5
        fisher_cross_short = fisher_prev[i] > 1.5 and fisher[i] <= 1.5
        
        # Fisher extreme values (for ranging regime)
        fisher_extreme_low = fisher[i] < -1.8
        fisher_extreme_high = fisher[i] > 1.8
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_ratio[i] > 1.2
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # REGIME 1: TRENDING (ADX > 25) - follow HTF trend with Fisher pullback
        if is_trending:
            # Long: HTF bull + Fisher crosses up from oversold + volume confirm
            if htf_1w_bull and fisher_cross_long and volume_confirmed:
                desired_signal = SIZE_STRONG
            
            # Short: HTF bear + Fisher crosses down from overbought + volume confirm
            elif htf_1w_bear and fisher_cross_short and volume_confirmed:
                desired_signal = -SIZE_STRONG
        
        # REGIME 2: RANGING (ADX < 20) - mean reversion with Fisher extremes
        elif is_ranging:
            # Long: Fisher extreme low + volume spike (reversal signal)
            if fisher_extreme_low and volume_confirmed:
                desired_signal = SIZE_BASE
            
            # Short: Fisher extreme high + volume spike (reversal signal)
            elif fisher_extreme_high and volume_confirmed:
                desired_signal = -SIZE_BASE
        
        # REGIME 3: TRANSITION (ADX 20-25) - reduced size, require stronger confirmation
        else:
            # Only enter if Fisher extreme + volume + HTF alignment
            if fisher_extreme_low and htf_1w_bull and volume_confirmed:
                desired_signal = SIZE_BASE * 0.8
            elif fisher_extreme_high and htf_1w_bear and volume_confirmed:
                desired_signal = -SIZE_BASE * 0.8
        
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
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif desired_signal >= SIZE_BASE * 0.7:
            final_signal = SIZE_BASE * 0.8
        elif desired_signal <= -SIZE_BASE * 0.7:
            final_signal = -SIZE_BASE * 0.8
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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