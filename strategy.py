#!/usr/bin/env python3
"""
4h Camarilla R3/S3 Breakout + 1d Williams %R Extreme + Volume Spike
Hypothesis: Camarilla R3/S3 levels represent stronger intraday support/resistance than R1/S1. 
Combined with 1d Williams %R extremes (oversold < -80, overbought > -20) and volume confirmation,
this captures institutional breakouts with higher conviction. Williams %R acts as a momentum filter
to avoid false breakouts in choppy markets. Works in bull/bear via extreme readings.
Target: 25-60 trades/year on 4h to stay within fee-efficient range.
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
    
    # Get 1d data for Camarilla pivots and Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (R3, S3) from 1d OHLC
    # Camarilla: R3 = close + 1.1*(high-low)/4, S3 = close - 1.1*(high-low)/4
    camarilla_r3 = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low']) / 4
    camarilla_s3 = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low']) / 4
    
    # Align Camarilla levels (no extra delay needed - pivots are based on completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    
    # Calculate 1d Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min()
    williams_r = (highest_high - df_1d['close']) / (highest_high - lowest_low) * -100
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(50).values  # neutral when undefined
    
    # Align Williams %R (no extra delay - it's based on completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
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
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(williams_r_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        williams_r_val = williams_r_aligned[i]
        atr_val = atr[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Williams %R extremes: oversold < -80, overbought > -20
        williams_oversold = williams_r_val < -80
        williams_overbought = williams_r_val > -20
        
        if position == 0:
            # Long: price breaks above R3 AND volume spike AND Williams oversold (momentum building)
            long_condition = (curr_high > r3) and volume_spike and williams_oversold
            # Short: price breaks below S3 AND volume spike AND Williams overbought (momentum building)
            short_condition = (curr_low < s3) and volume_spike and williams_overbought
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.5*ATR below entry) or Williams %R exits extreme (momentum fading)
            if curr_close <= entry_price - 2.5 * atr_val or williams_r_val > -50:  # exited oversold zone
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.5*ATR above entry) or Williams %R exits extreme (momentum fading)
            if curr_close >= entry_price + 2.5 * atr_val or williams_r_val < -50:  # exited overbought zone
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dWilliamsR_Extreme_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0