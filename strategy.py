#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with volume spike confirmation and 4h EMA50 trend filter
# Uses 4h/1d for signal direction (trend/structure) and 1h only for entry timing precision
# Volume spike (>2.0x 20-period MA) confirms institutional participation
# Session filter (08-20 UTC) reduces noise trades
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Novelty: Adapting proven 4h Camarilla strategy to 1h with strict filters to reduce fee drag

name = "1h_Camarilla_R3S3_VolumeSpike_4hEMA50_Trend_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate Camarilla pivot levels from previous 4h bar (more stable than 1d for 1h entries)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Previous 4h bar's OHLC for Camarilla calculation
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    prev_close = df_4h['close'].shift(1).values
    
    # Camarilla levels: R3/S3 = C ± (H-L)*1.1/4 (standard reversal bands)
    camarilla_range = (prev_high - prev_low) * 1.1 / 4.0
    r3 = prev_close + camarilla_range
    s3 = prev_close - camarilla_range
    
    # Align 4h levels to 1h timeframe (wait for 4h bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 50)  # warmup for volume MA and 4h EMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready or outside session
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema_50_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50 = ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation, session filter, and trend filter
            if curr_volume_confirm:
                # Bullish entry: price breaks above R3 with volume and above 4h EMA50
                if curr_high > curr_r3 and curr_close > curr_ema_50:
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below S3 with volume and below 4h EMA50
                elif curr_low < curr_s3 and curr_close < curr_ema_50:
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price breaks below S3 (reversal signal) or outside session
            if curr_low < curr_s3 or not in_session[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit when price breaks above R3 (reversal signal) or outside session
            if curr_high > curr_r3 or not in_session[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals