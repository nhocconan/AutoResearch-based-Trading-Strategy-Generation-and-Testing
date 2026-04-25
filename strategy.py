#!/usr/bin/env python3
"""
6h Weekly Pivot Breakout + 1d Volume Confirmation + 1d ADX Trend Filter
Hypothesis: Weekly Camarilla pivot levels (R4/S4) act as strong support/resistance. 
Breakouts above R4 or below S4 with volume confirmation and 1d ADX > 25 indicate 
institutional participation and trend continuation. Works in bull/bear by 
trading breakouts in direction of higher timeframe trend.
Target: 12-37 trades/year (50-150 over 4 years).
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
    
    # Get 1d data for weekly pivot calculation (need 5 days for prior week)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivots from prior week's OHLC
    # Weekly high = max(high) over last 5 trading days (approx)
    # Using rolling window of 5 days for weekly OHLC
    weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1).values  # prior week
    weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1).values
    weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().shift(1).values
    
    # Camarilla formulas for R4/S4
    # R4 = close + ((high - low) * 1.1 / 2)
    # S4 = close - ((high - low) * 1.1 / 2)
    weekly_range = weekly_high - weekly_low
    camarilla_r4 = weekly_close + (weekly_range * 1.1 / 2)
    camarilla_s4 = weekly_close - (weekly_range * 1.1 / 2)
    
    # Align weekly pivots to 6h timeframe (wait for weekly close)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4, additional_delay_bars=0)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4, additional_delay_bars=0)
    
    # Get 1d ADX for trend filter (ADX > 25 = trending)
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate ADX components
    plus_dm = pd.Series(df_1d['high']).diff()
    minus_dm = pd.Series(df_1d['low']).diff().multiply(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = pd.Series(df_1d['high']) - pd.Series(df_1d['low'])
    tr2 = abs(pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift(1))
    tr3 = abs(pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_values = adx.values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Get 1d volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for weekly pivot calculation (5 days) + ADX (14) + warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        r4_level = camarilla_r4_aligned[i]
        s4_level = camarilla_s4_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma = vol_ma_aligned[i]
        
        # Volume confirmation: current 6h volume > 1.5 * 1d average volume (scaled)
        # Approximate: 1d volume ≈ 4 * 6h volume (since 4x 6h bars in 1d)
        volume_spike = curr_volume > (1.5 * vol_ma / 4.0)
        
        # ADX trend filter: only trade when ADX > 25 (trending market)
        strong_trend = adx_val > 25
        
        if position == 0:
            # Long breakout: price closes above weekly R4 with volume and trend
            long_condition = (curr_close > r4_level) and volume_spike and strong_trend
            # Short breakout: price closes below weekly S4 with volume and trend
            short_condition = (curr_close < s4_level) and volume_spike and strong_trend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to weekly pivot point (mean of R4/S4) or trend weakens
            pivot_point = (r4_level + s4_level) / 2.0
            if curr_close <= pivot_point or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to weekly pivot point or trend weakens
            pivot_point = (r4_level + s4_level) / 2.0
            if curr_close >= pivot_point or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R4S4_Breakout_1dADX25_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0