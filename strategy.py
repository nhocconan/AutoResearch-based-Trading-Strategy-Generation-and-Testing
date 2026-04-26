#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeSpike_v1
Hypothesis: On 1h timeframe, Camarilla R1/S1 breakouts with 4h EMA50 trend filter and 1d volume spike (>2x 20-day average) provides high-probability directional signals with controlled trade frequency. Uses 4h for trend direction and 1d for volume confirmation to reduce false breakouts. 1h is used only for precise entry timing. Long when price > 4h EMA50 + breaks above R1 + 1d volume spike; short when price < 4h EMA50 + breaks below S1 + 1d volume spike. Uses discrete sizing (0.0, ±0.20) to minimize fee churn. Targets 60-150 total trades over 4 years (15-37/year) for optimal 1h frequency. Includes session filter (08-20 UTC) to avoid low-liquidity periods. Works in bull markets (trend following) and bear markets (trend following with short signals).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead and TypeError
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h data for HTF trend (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = pd.Series(df_4h['close'].values)
    ema_50_4h = close_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for HTF volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d 20-period volume average for spike confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 4h OHLC for Camarilla pivot levels (previous 4h bar)
    o_4h = df_4h['open'].values
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = c_4h + (h_4h - l_4h) * 1.1 / 12
    camarilla_s1 = c_4h - (h_4h - l_4h) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe (no additional delay needed)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50 warmup + volume MA
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            # Outside session: flatten position
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Volume spike: current 1h volume > 2x 1d 20-period average
        vol_spike = volume[i] > 2.0 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Long: price > 4h EMA50 + breaks above R1 + volume spike + session
            long_signal = (close[i] > ema_50_4h_aligned[i] and 
                          close[i] > camarilla_r1_aligned[i] and 
                          vol_spike)
            
            # Short: price < 4h EMA50 + breaks below S1 + volume spike + session
            short_signal = (close[i] < ema_50_4h_aligned[i] and 
                           close[i] < camarilla_s1_aligned[i] and 
                           vol_spike)
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: price closes below 4h EMA50 OR breaks below S1 (reversal)
            if close[i] < ema_50_4h_aligned[i] or close[i] < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: price closes above 4h EMA50 OR breaks above R1 (reversal)
            if close[i] > ema_50_4h_aligned[i] or close[i] > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeSpike_v1"
timeframe = "1h"
leverage = 1.0