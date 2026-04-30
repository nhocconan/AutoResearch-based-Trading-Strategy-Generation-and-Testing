#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation.
# Uses actual Camarilla pivot levels calculated from prior 4h bar (HLC).
# Long when price breaks above R3 with 4h uptrend (price > EMA50) and volume spike.
# Short when price breaks below S3 with 4h downtrend (price < EMA50) and volume spike.
# Exit when price retreats to R2/S2 level or opposite Camarilla level.
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
# Uses 4h/1d for signal direction, 1h only for entry timing precision.
# Session filter: 08-20 UTC to reduce noise trades.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_VolumeSpike_v1"
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
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    # Load 4h data ONCE before loop for Camarilla pivots and EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from prior 4h bar (HLC)
    # Typical Camarilla: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # where C = close, H = high, L = low of prior timeframe
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    camarilla_range = (high_4h - low_4h) * 1.1 / 2.0
    r3_4h = close_4h + camarilla_range
    s3_4h = close_4h - camarilla_range
    r2_4h = close_4h + camarilla_range * 2.0 / 3.0  # R2 = C + (H-L)*1.1/4
    s2_4h = close_4h - camarilla_range * 2.0 / 3.0  # S2 = C - (H-L)*1.1/4
    
    # Align Camarilla levels to 1h timeframe (wait for completed 4h bar)
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    r2_4h_aligned = align_htf_to_ltf(prices, df_4h, r2_4h)
    s2_4h_aligned = align_htf_to_ltf(prices, df_4h, s2_4h)
    
    # Volume confirmation: volume > 2.0x 24-bar average (1 day)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(r3_4h_aligned[i]) or np.isnan(s3_4h_aligned[i]) or
            np.isnan(r2_4h_aligned[i]) or np.isnan(s2_4h_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3, 4h uptrend, volume confirmation
            if (curr_close > r3_4h_aligned[i] and 
                curr_close > ema_50_4h_aligned[i] and 
                curr_volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3, 4h downtrend, volume confirmation
            elif (curr_close < s3_4h_aligned[i] and 
                  curr_close < ema_50_4h_aligned[i] and 
                  curr_volume_confirm):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:  # Long position
            # Exit conditions: price retreats to R2 or below S3 (reversal)
            if (curr_close <= r2_4h_aligned[i] or 
                curr_close < s3_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit conditions: price rallies to S2 or above R3 (reversal)
            if (curr_close >= s2_4h_aligned[i] or 
                curr_close > r3_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals