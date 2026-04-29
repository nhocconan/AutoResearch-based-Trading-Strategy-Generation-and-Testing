#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla levels provide precise intraday support/resistance; 1d EMA34 filters trend direction
# Volume spike confirms institutional participation. Works in bull/bear by trading with 1d trend
# at R3/S3 breakouts. Target: 12-25 trades/year (50-100 total).

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 (trend filter)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot points from previous 1d OHLC
    # Based on previous day's range: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4
    # S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    
    daily_range = prev_day_high - prev_day_low
    camarilla_r3 = prev_day_close + daily_range * 1.1 / 4
    camarilla_s3 = prev_day_close - daily_range * 1.1 / 4
    camarilla_r4 = prev_day_close + daily_range * 1.1 / 2
    camarilla_s4 = prev_day_close - daily_range * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (completed daily bar only)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: volume > 2.0x 20-period average (institutional participation)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34, 20)  # warmup for 1d EMA, Camarilla, volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if (np.isnan(ema_34_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_volume_spike = volume_spike[i]
        curr_ema_34 = ema_34_aligned[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_r4 = r4_aligned[i]
        curr_s4 = s4_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3 with volume spike + above 1d EMA34 (uptrend)
            if curr_close > curr_r3 and curr_volume_spike and curr_close > curr_ema_34:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume spike + below 1d EMA34 (downtrend)
            elif curr_close < curr_s3 and curr_volume_spike and curr_close < curr_ema_34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price breaks below S3 (failed breakout) OR reaches R4 (take profit)
            if curr_close < curr_s3 or curr_close > curr_r4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price breaks above R3 (failed breakdown) OR reaches S4 (take profit)
            if curr_close > curr_r3 or curr_close < curr_s4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals