# 12h_Camarilla_Pivot_R1S1_Breakout_VolumeSpike_1dTrend
# Hypothesis: Camarilla pivot R1/S1 levels on 12h chart act as strong support/resistance.
# Breakout above R1 with volume spike and 1d uptrend = long; breakdown below S1 with volume spike and 1d downtrend = short.
# Uses 1d EMA34 as trend filter to avoid counter-trend trades. Designed for low trade frequency (~15-25/year) with clear edge in both bull and bear markets via trend alignment.

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
    
    # Get 12h data for Camarilla pivot calculation (once before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla pivot levels for each 12h bar
    # Typical Price = (H + L + C) / 3
    typical_price = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    # Pivot Point = Typical Price
    pivot = typical_price.values
    # Range = High - Low
    range_hl = (df_12h['high'] - df_12h['low']).values
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    r1 = df_12h['close'].values + range_hl * 1.1 / 12
    s1 = df_12h['close'].values - range_hl * 1.1 / 12
    
    # Align to 12h timeframe (waits for bar close)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    
    # Get 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend direction
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (2x 20-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        above_r1 = price > r1_aligned[i]
        below_s1 = price < s1_aligned[i]
        above_1d_ema = price > ema_34_1d_aligned[i]
        below_1d_ema = price < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: break above R1 with volume spike and 1d uptrend
            if above_r1 and volume_spike[i] and above_1d_ema:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume spike and 1d downtrend
            elif below_s1 and volume_spike[i] and below_1d_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position: hold until break below S1 or trend change
            signals[i] = 0.25
            # Exit: price breaks below S1 or 1d trend turns down
            if below_s1 or below_1d_ema:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position: hold until break above R1 or trend change
            signals[i] = -0.25
            # Exit: price breaks above R1 or 1d trend turns up
            if above_r1 or above_1d_ema:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_Pivot_R1S1_Breakout_VolumeSpike_1dTrend"
timeframe = "12h"
leverage = 1.0