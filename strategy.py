#!/usr/bin/env python3
"""
1h_Camarilla_R3_S3_4hTrend_1dVolFilter_v1
Hypothesis: Use Camarilla R3/S3 breakout on 1h with 4h trend filter and 1d volume confirmation.
Long when price breaks above R3 with 4h uptrend (price > 4h EMA50) and 1d volume spike.
Short when price breaks below S3 with 4h downtrend (price < 4h EMA50) and 1d volume spike.
Camarilla levels provide institutional support/resistance; volume spike confirms institutional interest.
Trend filter avoids counter-trend trades. Works in bull/bear by aligning with 4h trend.
Target: 15-30 trades/year per symbol.
"""
name = "1h_Camarilla_R3_S3_4hTrend_1dVolFilter_v1"
timeframe = "1h"
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels for 1h (using previous bar's OHLC)
    # Camarilla: H-L range from previous bar
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    range_hl = prev_high - prev_low
    R3 = prev_close + range_hl * 1.1 / 2
    S3 = prev_close - range_hl * 1.1 / 2
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d volume filter: current 1h volume > 2.0 * 20-period average 1d volume (scaled)
    # Scale 1d volume to 1h equivalent: 1d volume / 6 (since 6*1h = 1d)
    vol_1h_equiv = df_1d['volume'].values / 6.0
    vol_avg_1d = pd.Series(vol_1h_equiv).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    volume_filter = volume > (vol_avg_1d_aligned * 2.0)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0
    
    start_idx = max(50, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_avg_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        # Require minimum 6 bars between trades to reduce frequency
        if bars_since_exit < 6:
            if position != 0:
                signals[i] = 0.20 if position == 1 else -0.20
            continue
        
        if position == 0:
            # Long: price breaks above R3 + 4h uptrend + volume spike + session
            if (close[i] > R3[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_filter[i] and 
                session_filter[i]):
                signals[i] = 0.20
                position = 1
                bars_since_exit = 0
            # Short: price breaks below S3 + 4h downtrend + volume spike + session
            elif (close[i] < S3[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_filter[i] and 
                  session_filter[i]):
                signals[i] = -0.20
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite Camarilla level or trend reversal
            if position == 1:
                if close[i] < S3[i] or close[i] < ema_50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_exit = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if close[i] > R3[i] or close[i] > ema_50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_exit = 0
                else:
                    signals[i] = -0.20
    
    return signals