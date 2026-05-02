#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA(34) trend filter and volume spike confirmation
# Camarilla pivot levels identify high-probability reversal/breakout zones; 1w EMA(34) > price for longs, < price for shorts filters primary trend
# Volume spike (2.0x 20-period average) ensures strong participation and reduces false breakouts
# Uses discrete position sizing 0.25 to minimize fee churn
# Targets 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by requiring volume confirmation and primary trend alignment

name = "12h_Camarilla_R3S3_Breakout_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(34) for trend filter
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels using previous 1d bar
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Camarilla alignment, EMA and volume MA)
    start_idx = 55  # max(20 for volume, 34 for EMA) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3 + price > 1w EMA + volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 + price < 1w EMA + volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price retreats to Camarilla R4 level (mean reversion target)
            camarilla_r4 = prev_close + (prev_high - prev_low) * 1.1 / 2
            camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
            if i < len(camarilla_r4_aligned) and not np.isnan(camarilla_r4_aligned[i]):
                if close[i] < camarilla_r4_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises to Camarilla S4 level (mean reversion target)
            camarilla_s4 = prev_close - (prev_high - prev_low) * 1.1 / 2
            camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
            if i < len(camarilla_s4_aligned) and not np.isnan(camarilla_s4_aligned[i]):
                if close[i] > camarilla_s4_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    return signals