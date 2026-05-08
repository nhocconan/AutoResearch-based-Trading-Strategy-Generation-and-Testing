#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla Pivot R1/S1 Breakout with 4h Trend and Volume Spike
# - Camarilla pivot levels from 4h: R1/S1 as breakout zones (tighter than R3/S3)
# - Breakout above R1 or below S1 with 4h trend alignment and volume spike
# - Uses 1h for precise entry timing, 4h for trend direction and pivot levels
# - Session filter (08-20 UTC) to avoid low-volume Asian session noise
# - Target: 15-35 trades/year to stay within fee limits (60-140 total over 4 years)

name = "1h_Camarilla_R1S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for Camarilla pivots and trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla pivot levels for each 4h bar
    # Pivot = (H + L + C) / 3
    # R1 = Pivot + (H - L) * 1.1 / 12
    # S1 = Pivot - (H - L) * 1.1 / 12
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    r1_4h = pivot_4h + (high_4h - low_4h) * 1.1 / 12.0
    s1_4h = pivot_4h - (high_4h - low_4h) * 1.1 / 12.0
    
    # Align Camarilla levels to 1h timeframe (wait for 4h bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # 4h EMA34 for trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    # Session filter: 08-20 UTC (avoid low-volume Asian session)
    # Pre-compute hours from DatetimeIndex to avoid type errors
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN or outside session
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(volume_spike[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + 4h uptrend + volume spike
            long_cond = (close[i] > r1_aligned[i] and 
                        ema_34_4h_aligned[i] > ema_34_4h_aligned[i-1] and
                        volume_spike[i])
            
            # Short: price breaks below S1 + 4h downtrend + volume spike
            short_cond = (close[i] < s1_aligned[i] and 
                         ema_34_4h_aligned[i] < ema_34_4h_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 (reversal signal)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above R1 (reversal signal)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals