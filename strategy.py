#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_VolumeSpike_ADXFilter
Hypothesis: Daily Camarilla pivot levels (R1, S1) act as strong intraday support/resistance.
Breakouts beyond these levels with volume confirmation and ADX trend filter capture momentum.
Works in bull/bear markets by requiring volume spike and ADX > 25 to ensure trending conditions.
Target: 20-50 trades per year on 4h timeframe to minimize fee drag.
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
    
    # Get daily data for Camarilla pivot calculation (once before loop)
    df_d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla pivot levels
    # Using previous day's data to avoid look-ahead
    daily_high = df_d['high']
    daily_low = df_d['low']
    daily_close = df_d['close']
    
    # Camarilla formulas
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R1 = Close + (Range * 1.1 / 12)
    # S1 = Close - (Range * 1.1 / 12)
    pivot = (daily_high + daily_low + daily_close) / 3
    daily_range = daily_high - daily_low
    r1 = daily_close + (daily_range * 1.1 / 12)
    s1 = daily_close - (daily_range * 1.1 / 12)
    
    # Shift by 1 to use previous day's levels only
    r1_prev = r1.shift(1).values
    s1_prev = s1.shift(1).values
    
    # Align to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_d, r1_prev)
    s1_aligned = align_htf_to_ltf(prices, df_d, s1_prev)
    
    # Volume spike: 2x 20-period average on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # ADX filter: 14-period ADX > 25 indicates trending market
    # Calculate +DI and -DI
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    # Pad arrays to match original length
    plus_dm_padded = np.concatenate([[np.nan], plus_dm])
    minus_dm_padded = np.concatenate([[np.nan], minus_dm])
    tr_padded = np.concatenate([[np.nan], tr])
    
    atr_period = 14
    atr_values = wilders_smooth(tr_padded, atr_period)
    plus_di_values = wilders_smooth(plus_dm_padded, atr_period)
    minus_di_values = wilders_smooth(minus_dm_padded, atr_period)
    
    # Avoid division by zero
    dx = np.where((plus_di_values + minus_di_values) != 0,
                  100 * np.abs(plus_di_values - minus_di_values) / (plus_di_values + minus_di_values),
                  0)
    adx_values = wilders_smooth(dx, atr_period)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_d, adx_values)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        adx_val = adx_aligned[i]
        
        if position == 0:
            # Long: break above R1 with volume spike and ADX > 25 (trending up)
            if price > r1_val and volume_spike[i] and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume spike and ADX > 25 (trending down)
            elif price < s1_val and volume_spike[i] and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price returns to S1 or ADX drops below 20 (trend weakening)
            if price <= s1_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price returns to R1 or ADX drops below 20 (trend weakening)
            if price >= r1_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_VolumeSpike_ADXFilter"
timeframe = "4h"
leverage = 1.0