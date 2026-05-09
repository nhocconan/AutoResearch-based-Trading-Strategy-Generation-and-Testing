# 1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeS
# 1d timeframe, 1w HTF trend filter
# Long when price breaks above R1 with 1w EMA50 uptrend and volume spike
# Short when price breaks below S1 with 1w EMA50 downtrend and volume spike
# Exit when price returns to central pivot (PP)
# Uses weekly EMA for trend to avoid overtrading, volume for conviction
# Target: 30-80 total trades over 4 years (7-20/year) with size 0.25
# Designed for both bull and bear markets via trend filter and volume confirmation

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeS"
timeframe = "1d"
leverage = 1.0

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
    
    # Calculate 1d Camarilla levels from previous day
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    prev_close = df_1d['close'].shift(1)
    
    # Pivot point and Camarilla levels
    pp = (prev_high + prev_low + prev_close) / 3
    r1 = pp + (prev_high - prev_low) * 1.0833
    r2 = pp + (prev_high - prev_low) * 1.1666
    r3 = pp + (prev_high - prev_low) * 1.2500
    s1 = pp - (prev_high - prev_low) * 1.0833
    s2 = pp - (prev_high - prev_low) * 1.1666
    s3 = pp - (prev_high - prev_low) * 1.2500
    
    # Align to 1d timeframe (same as prices, but ensure proper alignment)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 2x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA and volume calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R1, 1w EMA50 uptrend, volume spike
            if (close[i] > r1_aligned[i] and 
                ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1, 1w EMA50 downtrend, volume spike
            elif (close[i] < s1_aligned[i] and 
                  ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to central pivot
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to central pivot
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals