# 1h_Camarilla_R3_S3_Breakout_4hTrend_Volume
# Hypothesis: Use 4h Camarilla R3/S3 breakouts as signal direction with 1h for precise entry timing.
# Filter by 4h EMA34 trend and volume spikes. Target 15-37 trades/year to minimize fee drag.
# Works in bull/bear markets by following 4h trend direction.
# Session filter (08-20 UTC) reduces noise during low-volume hours.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h data for trend and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h EMA34 for trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 4h Camarilla levels (R3, S3)
    n4h = len(close_4h)
    camarilla_R3_4h = np.full(n4h, np.nan)
    camarilla_S3_4h = np.full(n4h, np.nan)
    
    for i in range(1, n4h):
        PH = high_4h[i-1]
        PL = low_4h[i-1]
        PC = close_4h[i-1]
        
        R3 = PC + 1.1 * (PH - PL)
        S3 = PC - 1.1 * (PH - PL)
        
        camarilla_R3_4h[i] = R3
        camarilla_S3_4h[i] = S3
    
    camarilla_R3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_R3_4h)
    camarilla_S3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_S3_4h)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN or outside session
        if (not in_session[i] or 
            np.isnan(camarilla_R3_4h_aligned[i]) or np.isnan(camarilla_S3_4h_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with 4h uptrend + volume spike
            long_cond = (close[i] > camarilla_R3_4h_aligned[i] and 
                        ema_34_4h_aligned[i] > ema_34_4h_aligned[i-1] and
                        volume_spike[i])
            
            # Short: price breaks below S3 with 4h downtrend + volume spike
            short_cond = (close[i] < camarilla_S3_4h_aligned[i] and 
                         ema_34_4h_aligned[i] < ema_34_4h_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 (reversion to mean)
            if close[i] < camarilla_S3_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above R3 (reversion to mean)
            if close[i] > camarilla_R3_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals