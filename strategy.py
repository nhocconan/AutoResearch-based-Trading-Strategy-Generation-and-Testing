#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1d_EMA34_Trend_VolumeChop_v1
Hypothesis: On 12h timeframe, trade Camarilla R1/S1 breakouts from previous 1d bar only when aligned with 1d EMA34 trend, confirmed by volume spike (>2.0x 20-bar average), and in non-choppy markets (Choppiness Index < 61.8). Uses ATR trailing stop. Designed for lower trade frequency (~12-37/year) to minimize fee drag while capturing institutional breakouts in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla R1 and S1 from previous 1d bar (HLC of daily)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # Camarilla R1 = close + ((high - low) * 1.1 / 12)
    # Camarilla S1 = close - ((high - low) * 1.1 / 12)
    camarilla_r1_1d = close_1d_arr + ((high_1d - low_1d) * 1.1 / 12)
    camarilla_s1_1d = close_1d_arr - ((high_1d - low_1d) * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    
    # ATR for stoploss and volatility
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Choppiness Index ( choppy > 61.8, trending < 38.2 )
    chop_period = 14
    # True Range
    tr_chop = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_chop[0] = high[0] - low[0]
    atr_chop = pd.Series(tr_chop).ewm(span=chop_period, min_periods=chop_period, adjust=False).mean()
    # Highest high and lowest low over chop_period
    hh = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max()
    ll = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min()
    # Chop = 100 * log10(sum(TR) / (HH - LL)) / log10(chop_period)
    sum_tr = pd.Series(tr_chop).rolling(window=chop_period, min_periods=chop_period).sum()
    chop = 100 * np.log10(sum_tr / (hh - ll)) / np.log10(chop_period)
    chop = chop.values
    chop_filter = chop < 61.8  # non-choppy (trending) market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of EMA34 (34), ATR (14), volume MA (20), chop (14)
    start_idx = max(34, 14, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_34_val = ema_34_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        r1_level = camarilla_r1_aligned[i]
        s1_level = camarilla_s1_aligned[i]
        is_chop = chop_filter[i]
        
        if position == 0:
            # Long: Break above R1, above 1d EMA34, volume spike, non-choppy
            long_signal = (high_val > r1_level) and (close_val > ema_34_val) and vol_spike and is_chop
            
            # Short: Break below S1, below 1d EMA34, volume spike, non-choppy
            short_signal = (low_val < s1_level) and (close_val < ema_34_val) and vol_spike and is_chop
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, close_val)
            # Exit: Close below EMA34 (trend change) OR trailing stop (2.5*ATR below high)
            if (close_val < ema_34_val) or (close_val < highest_since_entry - 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, close_val)
            # Exit: Close above EMA34 (trend change) OR trailing stop (2.5*ATR above low)
            if (close_val > ema_34_val) or (close_val > lowest_since_entry + 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1d_EMA34_Trend_VolumeChop_v1"
timeframe = "12h"
leverage = 1.0