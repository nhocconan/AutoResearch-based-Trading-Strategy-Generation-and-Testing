#!/usr/bin/env python3
"""
6h Camarilla Pivot Breakout + 12h EMA50 Trend + Volume Confirmation
Hypothesis: Camarilla R3/S3 levels from 12h chart act as significant intraday support/resistance.
Breakouts beyond these levels with volume confirmation and 12h EMA50 trend filter capture
institutional flow with controlled frequency. Works in bull/bear via trend filter.
Target: 12-37 trades/year on 6h.
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
    
    # Get 12h data for Camarilla pivots and EMA50 trend
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels on 12h
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    # Calculate pivot levels
    camarilla_r4 = c_12h + ((h_12h - l_12h) * 1.1 / 2)
    camarilla_r3 = c_12h + ((h_12h - l_12h) * 1.1 / 4)
    camarilla_s3 = c_12h - ((h_12h - l_12h) * 1.1 / 4)
    camarilla_s4 = c_12h - ((h_12h - l_12h) * 1.1 / 2)
    
    # Align Camarilla levels with 1-bar delay (wait for 12h bar close)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR(14) for stoploss
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for data to propagate
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_12h_aligned[i]) or 
            np.isnan(s3_12h_aligned[i]) or 
            np.isnan(r4_12h_aligned[i]) or 
            np.isnan(s4_12h_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        r3 = r3_12h_aligned[i]
        s3 = s3_12h_aligned[i]
        r4 = r4_12h_aligned[i]
        s4 = s4_12h_aligned[i]
        ema_50 = ema_50_12h_aligned[i]
        atr_val = atr[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Trend filter
        uptrend = curr_close > ema_50
        downtrend = curr_close < ema_50
        
        if position == 0:
            # Long: price breaks above R3 with volume spike AND uptrend
            long_condition = (curr_high > r3) and volume_spike and uptrend
            # Short: price breaks below S3 with volume spike AND downtrend
            short_condition = (curr_low < s3) and volume_spike and downtrend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.5*ATR below entry) or trend reversal
            if curr_close <= entry_price - 2.5 * atr_val or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.5*ATR above entry) or trend reversal
            if curr_close >= entry_price + 2.5 * atr_val or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0