#!/usr/bin/env python3
"""
Experiment #021: 4h Donchian Breakout + Choppiness Regime + Volume

HYPOTHESIS: Donchian(20) breakouts capture institutional momentum.
By adding choppiness regime (CHOP > 61.8 = range → skip, CHOP < 50 = trending → trade),
we avoid false breakouts in choppy markets that destroyed 2022 performance.
4h timeframe should generate 75-150 trades over 4 years.

WHY IT WORKS IN BULL AND BEAR:
- Bull: Price breaks above Donchian upper → long
- Bear: Price breaks below Donchian lower → short
- Choppiness filter avoids whipsaws in 2022 crash/range
- Volume confirms institutional conviction

TARGET: 75-150 total trades over 4 years (19-37/year). HARD MAX: 200.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_chop_vol_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP): 
    - CHOP > 61.8 = choppy/range market (avoid)
    - CHOP < 50 = trending market (trade)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high - lowest_low > 1e-10:
            atr_sum = np.sum(np.abs(np.diff(close[i-period:i+1], prepend=close[i-period])))
            # Use simple ATR sum instead
            tr_sum = 0.0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                tr_sum += tr
            
            chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period + 1)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA21 for trend (faster than 50, less lag than 200)
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness_index(high, low, close, period=14)
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Donchian channels (20 periods = 5 days at 4h)
    donchian_period = 20
    upper_donchian = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_donchian = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    middle_donchian = (upper_donchian + lower_donchian) / 2
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = max(50, donchian_period)  # Need Donchian to be ready
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME FILTER: Only trade when trending (CHOP < 50) ===
        # Skip entirely in choppy market
        chop_trending = chop[i] < 50.0
        
        # === TREND DIRECTION (1d EMA21) ===
        bull_trend = close[i] > ema_1d_aligned[i]
        bear_trend = close[i] < ema_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # Donchian breakout signals
        upper_break = high[i] > upper_donchian[i - 1]  # Previous bar's upper
        lower_break = low[i] < lower_donchian[i - 1]   # Previous bar's lower
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Upper breakout + bull trend + chop trending + volume ===
            if upper_break and bull_trend and chop_trending and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: Lower breakout + bear trend + chop trending + volume ===
            if lower_break and bear_trend and chop_trending and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars (12h) to avoid fee churn ===
        bars_held = i - entry_bar
        if in_position and bars_held >= 3:
            # Exit if price returns to middle Donchian
            if position_side > 0 and close[i] < middle_donchian[i - 1]:
                desired_signal = 0.0
            if position_side < 0 and close[i] > middle_donchian[i - 1]:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = low[i] - 2.5 * entry_atr
                else:
                    stop_price = high[i] + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals