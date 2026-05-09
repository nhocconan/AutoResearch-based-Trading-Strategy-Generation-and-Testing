# 12h_Camarilla_R1_S1_Breakout_1wEMA50_Trend
# Hypothesis: 12-hour Camarilla R1/S1 breakout with weekly EMA50 trend filter
# Long when price breaks above R1 with weekly EMA50 uptrend and volume spike
# Short when price breaks below S1 with weekly EMA50 downtrend and volume spike
# Exit when price retouches central pivot (PP) or reverses to opposite S4/R4
# Uses 12h timeframe to reduce trade frequency and avoid fee drag
# Weekly EMA50 provides strong trend filter for both bull and bear markets
# Designed for 50-150 total trades over 4 years (12-37/year) with size 0.25

name = "12h_Camarilla_R1_S1_Breakout_1wEMA50_Trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w Camarilla levels (PP, R1, R2, R3, R4, S1, S2, S3, S4)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Previous week's OHLC for Camarilla calculation
    prev_high = df_1w['high'].shift(1)
    prev_low = df_1w['low'].shift(1)
    prev_close = df_1w['close'].shift(1)
    
    # Calculate pivot point
    pp = (prev_high + prev_low + prev_close) / 3
    # Calculate Camarilla levels
    r1 = pp + (prev_high - prev_low) * 1.0833
    r2 = pp + (prev_high - prev_low) * 1.1666
    r3 = pp + (prev_high - prev_low) * 1.2500
    r4 = pp + (prev_high - prev_low) * 1.5000
    s1 = pp - (prev_high - prev_low) * 1.0833
    s2 = pp - (prev_high - prev_low) * 1.1666
    s3 = pp - (prev_high - prev_low) * 1.2500
    s4 = pp - (prev_high - prev_low) * 1.5000
    
    # Align Camarilla levels to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp.values)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1.values)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4.values)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4.values)
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for EMA calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R1, EMA50 uptrend, volume spike
            if (close[i] > r1_aligned[i] and 
                ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1, EMA50 downtrend, volume spike
            elif (close[i] < s1_aligned[i] and 
                  ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price retouches central pivot or reverses to S4
            if (close[i] <= pp_aligned[i]) or (close[i] < s4_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price retouches central pivot or reverses to R4
            if (close[i] >= pp_aligned[i]) or (close[i] > r4_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals