#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA34 trend filter and 1d volume spike
# Camarilla R3/S3 levels provide high-probability reversal/breakout zones from intraday structure
# 12h EMA34 filter ensures we only trade in the direction of the intermediate trend
# 1d volume spike confirms institutional participation, reducing false breakouts
# Designed for low frequency (50-150 trades over 4 years) with discrete sizing
# Works in both bull and bear: volume confirms legitimacy, trend filter avoids counter-trend whipsaws

name = "6h_Camarilla_R3S3_Breakout_12hEMA34_1dVolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # 1d HTF data for volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1d session
    # Using 1d high/low/close from prior completed day
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Camarilla levels: based on prior day's range
    # R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    camarilla_range = (high_1d - low_1d) * 1.1 / 2
    r3 = close_1d + camarilla_range
    s3 = close_1d - camarilla_range
    r4 = close_1d + camarilla_range * 2  # R4 = close + 1.1*(high-low)
    s4 = close_1d - camarilla_range * 2  # S4 = close - 1.1*(high-low)
    
    # Align Camarilla levels to 6h timeframe (using completed 1d bars)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # 1d volume spike: volume > 2.0 * 20-period EMA
    vol_1d = df_1d['volume'].values
    vol_ema_20_1d = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20_1d)
    volume_spike = volume > (2.0 * vol_ema_20_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(40, 34)  # Need EMA and volume EMA
    
    for i in range(start_idx, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_34_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade long when price > EMA34, short when price < EMA34
        bullish_trend = close[i] > ema_34_aligned[i]
        bearish_trend = close[i] < ema_34_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above R3 with volume spike in bullish trend
            if bullish_trend and close[i] > r3_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 with volume spike in bearish trend
            elif bearish_trend and close[i] < s3_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price returns to S3 or breaks S4 (failed breakout)
            if close[i] <= s3_aligned[i] or close[i] < s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price returns to R3 or breaks R4 (failed breakout)
            if close[i] >= r3_aligned[i] or close[i] > r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals