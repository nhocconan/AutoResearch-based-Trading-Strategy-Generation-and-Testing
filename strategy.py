#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4H long/short at 1D Camarilla S1/R1 with 1D EMA34 trend filter and volume spike.
# Uses Camarilla pivot levels as dynamic support/resistance and EMA for trend direction.
# Designed to work in both bull (pullbacks to S1) and bear (rejections at R1).
# Target: 20-40 trades/year to avoid fee drag.
name = "4H_Camarilla_S1R1_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for 1-day
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = df_1d['close'] + (df_1d['high'] - df_1d['low']) * 1.1 / 12.0
    camarilla_s1 = df_1d['close'] - (df_1d['high'] - df_1d['low']) * 1.1 / 12.0
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Camarilla levels and EMA to 4h timeframe (use previous day's values)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 20-period EMA (high threshold for fewer trades)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least 1 day of data for Camarilla/EMA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: price crosses above S1 + 1d uptrend + volume spike
            if (price > camarilla_s1_aligned[i] and price > ema_34_1d_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price crosses below R1 + 1d downtrend + volume spike
            elif (price < camarilla_r1_aligned[i] and price < ema_34_1d_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below S1 or trend reverses
            if price < camarilla_s1_aligned[i] or price < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above R1 or trend reverses
            if price > camarilla_r1_aligned[i] or price > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals