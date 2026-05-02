#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA(34) trend filter and volume spike confirmation
# Camarilla pivot levels identify key intraday support/resistance; breakout above R3 or below S3 with volume
# and 1d EMA trend alignment captures strong momentum moves. Works in bull/bear via volume confirmation
# and primary trend filter. Discrete sizing 0.25 to minimize fee churn. Targets 20-30 trades/year.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1d Camarilla levels (R3, S3) based on previous day's OHLC
    # Need to shift OHLC by 1 to get previous day's values
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla R3 and S3
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.1 * camarilla_range
    s3 = prev_close - 1.1 * camarilla_range
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3, additional_delay_bars=0)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3, additional_delay_bars=0)
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Camarilla calculation, EMA and volume MA)
    start_idx = 55  # max(20 for volume MA, 34 for EMA) + buffer for 1d data shift
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3 + price > 1d EMA + volume spike
            if close[i] > r3_aligned[i] and close[i] > ema_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 + price < 1d EMA + volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price retreats to midpoint between R3 and S3 (which is the prior day's close)
            # Midpoint of R3 and S3 is actually the prior day's close
            if close[i] < prev_close_aligned[i]:  # We need to align prev_close
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises to midpoint between R3 and S3 (prior day's close)
            if close[i] > prev_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals