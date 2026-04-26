#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeConfirmation_v2
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Uses 4h timeframe for balanced trade frequency (target: 75-200 total trades over 4 years)
- Camarilla R1/S1 levels from 1d provide precise support/resistance from prior day
- 1d EMA34 filter ensures trades align with higher timeframe trend
- Volume spike (2.0x 20-period average) confirms institutional participation
- Designed for 20-50 trades/year (80-200 total over 4 years) to minimize fee drag
- Works in bull/bear markets by trading with the 1d trend and using volume/ADX to filter false breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate ADX(14) on 1d for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values (Wilder's smoothing)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: smoothed = (prev_smoothed * (period-1) + current) / period
            for i in range(period, len(data)):
                if not np.isnan(data[i]) and not np.isnan(result[i-1]):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_smooth = wilder_smooth(tr, 14)
    plus_dm_smooth = wilder_smooth(plus_dm, 14)
    minus_dm_smooth = wilder_smooth(minus_dm, 14)
    
    # DI+ and DI-
    plus_di = np.where(tr_smooth != 0, (plus_dm_smooth / tr_smooth) * 100, 0)
    minus_di = np.where(tr_smooth != 0, (minus_dm_smooth / tr_smooth) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = wilder_smooth(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Camarilla levels from previous 1d bar
    camarilla_r1 = close_1d + (1.0/12) * (high_1d - low_1d)
    camarilla_s1 = close_1d - (1.0/12) * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume spike: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 1d EMA/ADX, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(adx_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions
        breakout_long = close[i] > camarilla_r1_aligned[i]
        breakout_short = close[i] < camarilla_s1_aligned[i]
        
        # Re-entry conditions (price back inside Camarilla H3-L3 range)
        # Calculate Camarilla H3/L3 for the day: H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low)
        camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d)
        camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d)
        camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
        camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
        price_in_range = (close[i] > camarilla_l3_aligned[i]) and (close[i] < camarilla_h3_aligned[i])
        
        if position == 0:
            # Long: breakout above R1 AND close > 1d EMA34 AND volume spike AND ADX > 25
            if breakout_long and close[i] > ema34_1d_aligned[i] and volume_spike[i] and adx_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 AND close < 1d EMA34 AND volume spike AND ADX > 25
            elif breakout_short and close[i] < ema34_1d_aligned[i] and volume_spike[i] and adx_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: breakout below S1 OR price re-enters Camarilla H3-L3 range
            if breakout_short or price_in_range:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: breakout above R1 OR price re-enters Camarilla H3-L3 range
            if breakout_long or price_in_range:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeConfirmation_v2"
timeframe = "4h"
leverage = 1.0