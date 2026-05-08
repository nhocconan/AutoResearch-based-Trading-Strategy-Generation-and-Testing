# 1d Camarilla R3/S3 Breakout with 1w Trend and Volume Spike
# - Uses Camarilla levels from weekly timeframe (S3/S2 for long, R3/R2 for short)
# - Breakout above S3 with 1w uptrend or below R3 with 1w downtrend
# - Volume spike confirms breakout strength
# - Works in bull/bear by using 1w trend filter to avoid counter-trend trades
# - Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag on 1d timeframe

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels using previous week's data
    # S2 = C - (H-L)*1.16, S3 = C - (H-L)*1.25, R2 = C + (H-L)*1.16, R3 = C + (H-L)*1.25
    n1w = len(close_1w)
    camarilla_S2 = np.full(n1w, np.nan)
    camarilla_S3 = np.full(n1w, np.nan)
    camarilla_R2 = np.full(n1w, np.nan)
    camarilla_R3 = np.full(n1w, np.nan)
    
    for i in range(1, n1w):
        H = high_1w[i-1]
        L = low_1w[i-1]
        C = close_1w[i-1]
        range_val = H - L
        camarilla_S2[i] = C - range_val * 1.16
        camarilla_S3[i] = C - range_val * 1.25
        camarilla_R2[i] = C + range_val * 1.16
        camarilla_R3[i] = C + range_val * 1.25
    
    # Align Camarilla levels to 1d timeframe
    camarilla_S2_aligned = align_htf_to_ltf(prices, df_1w, camarilla_S2)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_S3)
    camarilla_R2_aligned = align_htf_to_ltf(prices, df_1w, camarilla_R2)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_R3)
    
    # 1w data for trend filter
    close_1w_close = df_1w['close'].values
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_S2_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or 
            np.isnan(camarilla_R2_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above S3 (strong support) with 1w uptrend + volume spike
            long_cond = (close[i] > camarilla_S3_aligned[i] and 
                        ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1] and
                        volume_spike[i])
            
            # Short: price breaks below R3 (strong resistance) with 1w downtrend + volume spike
            short_cond = (close[i] < camarilla_R3_aligned[i] and 
                         ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S2 (weaker support)
            if close[i] < camarilla_S2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R2 (weaker resistance)
            if close[i] > camarilla_R2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals