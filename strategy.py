#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot (R3/S3) breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla pivots identify key support/resistance levels. Breakouts above R3 or below S3 with volume
# spike and aligned 1d trend capture strong momentum moves. Designed for 12-37 trades/year to minimize
# fee drag. Works in bull markets via upside breakouts and in bear markets via downside breakdowns.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation and EMA34 - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (R3, S3) on 1d data
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    #          S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    # We use R3 and S3 for breakout signals
    daily_range = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * daily_range * 1.1 / 4
    camarilla_s3 = close_1d - 1.1 * daily_range * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 1d data for EMA34 trend filter - ONCE before loop
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 12h timeframe (wait for completed 1d bar)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate volume spike filter (20-period volume MA)
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Close breaks above camarilla R3 AND volume spike AND 1d EMA34 uptrend
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Close breaks below camarilla S3 AND volume spike AND 1d EMA34 downtrend
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close drops below camarilla S3 (reversal) OR trend reverses
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close rises above camarilla R3 (reversal) OR trend reverses
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals