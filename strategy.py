#!/usr/bin/env python3
"""
Experiment #027: 6h Williams Alligator Pullback + ADX Trend

HYPOTHESIS: Williams Alligator (3 smoothed EMAs at different speeds) identifies
institutional trend direction. Pullbacks TO the Alligator jaw (slowest line) in
the direction of the main trend represent low-risk entries. ADX confirms trend
strength and prevents trading in chop. This is a trend-following pullback strategy
that captures "fade the move" entries with tight stops.

WHY 6h: Slower than 4h (reduces fee drag), but fast enough to catch institutional
moves that take 1-3 days to develop. The Alligator's 13-period jaw (≈3 days on 6h)
aligns well with swing trading timeframes.

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Long entries: Pullback TO jaw during uptrend (jaw above teeth above lips)
- Short entries: Rally TO jaw during downtrend (jaw below teeth below lips)
- Symmetric logic captures both directions
- ADX filter keeps us out of choppy periods
- 2022 crash creates short opportunities, 2021/2023 rallies create longs

TARGET: 75-150 total trades over 4 years = 18-37/year.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_alligator_adx_pullback_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ema(close, period, smooth=1):
    """Williams Alligator uses SMA for initial, then EMA-like smoothing"""
    if smooth == 1:
        return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    else:
        # Use SMA as base (Williams method)
        sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
        return sma

def calculate_alligator(high, low, close):
    """
    Williams Alligator:
    - JAW (blue): 13-period SMA, smoothed (Williams = 8 bars offset)
    - TEETH (red): 8-period SMA, smoothed (Williams = 5 bars offset)  
    - LIPS (green): 5-period SMA, smoothed (Williams = 3 bars offset)
    
    Uptrend: JAW > TEETH > LIPS
    Downtrend: JAW < TEETH < LIPS
    """
    n = len(close)
    
    # Williams Alligator uses SMMA (Smoothed Moving Average)
    # SMMA[i] = (SMMA[i-1] * (period-1) + close[i]) / period
    def smma(series, period):
        result = np.zeros(len(series), dtype=np.float64)
        result[0] = series[0]
        for i in range(1, len(series)):
            result[i] = (result[i-1] * (period - 1) + series[i]) / period
        return result
    
    # Median price as base
    median = (high + low) / 2
    
    jaw = smma(median, 13)    # Slowest - trend line
    teeth = smma(median, 8)   # Medium
    lips = smma(median, 5)    # Fastest - signal line
    
    return jaw, teeth, lips

def calculate_adx_dmi(high, low, close, period=14):
    """Calculate ADX and DMI components"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth with Wilder's smoothing
    def wilder_smooth(values, period):
        result = np.zeros(len(values), dtype=np.float64)
        result[period] = np.sum(values[1:period+1])
        for i in range(period + 1, len(values)):
            result[i] = result[i-1] - result[i-1] / period + values[i]
        return result
    
    atr_smooth = wilder_smooth(tr, period)
    plus_dm_smooth = wilder_smooth(plus_dm, period)
    minus_dm_smooth = wilder_smooth(minus_dm, period)
    
    # Calculate DI
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        if atr_smooth[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / atr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / atr_smooth[i]
    
    # DX
    dx = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX = Wilder smooth of DX
    adx = wilder_smooth(dx, period)
    
    return adx, plus_di, minus_di, atr_smooth

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Wilder smoothing
    atr = np.zeros(n, dtype=np.float64)
    atr[period] = np.sum(tr[1:period+1])
    for i in range(period + 1, n):
        atr[i] = atr[i-1] - atr[i-1] / period + tr[i]
    
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA50 for medium-term trend
    sma50_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    # === Local 6h indicators ===
    jaw, teeth, lips = calculate_alligator(high, low, close)
    adx, plus_di, minus_di, atr_raw = calculate_adx_dmi(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30  # Standard sizing
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 100  # Alligator needs ~13+ buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(jaw[i]) or np.isnan(adx[i]) or np.isnan(atr_14[i]) or atr_14[i] <= 0:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma50_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DETECTION (Alligator) ===
        # Uptrend: jaw > teeth > lips (all aligned up)
        # Downtrend: jaw < teeth < lips (all aligned down)
        j = jaw[i]
        t = teeth[i]
        l = lips[i]
        
        is_uptrend = j > t > l
        is_downtrend = j < t < l
        
        # === ADX CONFIRMATION ===
        adx_val = adx[i]
        pdi = plus_di[i]
        mdi = minus_di[i]
        
        # ADX > 22 = trending (not choppy)
        # For longs: ADX rising AND +DI > -DI
        # For shorts: ADX rising AND -DI > +DI
        adx_trending = adx_val > 22
        
        # === PULLBACK DETECTION ===
        # Price pulled back to jaw after initial move
        price_vs_jaw = close[i] - j
        price_vs_teeth = close[i] - t
        
        # === DI CROSS FOR ENTRY TIMING ===
        # +DI crossed above -DI = bullish momentum building
        # -DI crossed above +DI = bearish momentum building
        pdi_prev = plus_di[i-1] if i > 0 else pdi
        mdi_prev = minus_di[i-1] if i > 0 else mdi
        
        pdi_cross_up = pdi > mdi and pdi_prev <= mdi_prev  # Bullish cross
        pdi_cross_down = mdi > pdi and mdi_prev <= pdi_prev  # Bearish cross
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.2
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY: Uptrend pullback to jaw ===
            # Conditions:
            # 1. Alligator shows uptrend (jaw > teeth > lips)
            # 2. Price has pulled back to near jaw
            # 3. +DI crossed above -DI (momentum confirmation)
            # 4. ADX confirms trend strength
            if is_uptrend:
                # Price close to jaw (within 1 ATR = pullback)
                pullback_tolerance = 1.5 * atr_14[i]
                
                if abs(price_vs_jaw) < pullback_tolerance:
                    if pdi_cross_up and adx_trending:
                        if vol_spike:
                            desired_signal = SIZE
            
            # === SHORT ENTRY: Downtrend rally to jaw ===
            # Conditions:
            # 1. Alligator shows downtrend (jaw < teeth < lips)
            # 2. Price has rallied to near jaw
            # 3. -DI crossed above +DI (momentum confirmation)
            # 4. ADX confirms trend strength
            if is_downtrend:
                pullback_tolerance = 1.5 * atr_14[i]
                
                if abs(price_vs_jaw) < pullback_tolerance:
                    if pdi_cross_down and adx_trending:
                        if vol_spike:
                            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TIME-BASED EXIT (hold at least 6 bars = 1.5 days) ===
        if in_position:
            bars_held = i - entry_bar
            min_hold = 6
            
            if bars_held >= min_hold:
                # Exit if trend reverses (Alligator flip)
                if position_side > 0:
                    # Exit long if jaw crosses below teeth
                    jaw_prev = jaw[i-1]
                    teeth_prev = teeth[i-1]
                    if jaw < teeth and jaw_prev >= teeth_prev:
                        desired_signal = 0.0
                
                if position_side < 0:
                    jaw_prev = jaw[i-1]
                    teeth_prev = teeth[i-1]
                    if jaw > teeth and jaw_prev <= teeth_prev:
                        desired_signal = 0.0
        
        # === ADX EXIT FILTER (stop if trend weakens) ===
        if in_position:
            # Exit if ADX drops below 18 (trend exhausted)
            if adx_val < 18:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals