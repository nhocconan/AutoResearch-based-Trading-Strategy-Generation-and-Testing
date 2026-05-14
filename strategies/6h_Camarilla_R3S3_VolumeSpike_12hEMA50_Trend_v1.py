#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with volume spike confirmation and 12h EMA50 trend filter
# Uses proven Camarilla pivot structure (R3/S3 = standard reversal levels) with 12h EMA50 trend filter
# Volume spike (>2.0x 20-period MA) confirms institutional participation
# 6h timeframe balances responsiveness with lower fee drag vs lower timeframes
# Works in bull/bear: volume confirms breakout validity, 12h EMA50 filters counter-trend noise
# Target: 75-150 total trades over 4 years (19-37/year) for 6h timeframe
# Novelty: Applying proven Camarilla+Volume+Trend framework to 6h timeframe with 12h HTF filter

name = "6h_Camarilla_R3S3_VolumeSpike_12hEMA50_Trend_v1"
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
    
    # Calculate Camarilla pivot levels from previous day
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels: R3/S3 = C ± (H-L)*1.1/4 (standard bands)
    camarilla_range = (prev_high - prev_low) * 1.1 / 4.0
    r3 = prev_close + camarilla_range
    s3 = prev_close - camarilla_range
    
    # Align daily levels to 6h timeframe (wait for daily bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # 12h EMA50 for trend filter (HTF = 12h as requested)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 50)  # warmup for volume MA and 12h EMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(ema_50_aligned[i]):
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
            # Only trade with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish entry: price breaks above R3 with volume and above 12h EMA50
                if curr_high > curr_r3 and curr_close > curr_ema_50:
                    signals[i] = 0.30
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below S3 with volume and below 12h EMA50
                elif curr_low < curr_s3 and curr_close < curr_ema_50:
                    signals[i] = -0.30
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price breaks below S3 (reversal signal)
            if curr_low < curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit when price breaks above R3 (reversal signal)
            if curr_high > curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals