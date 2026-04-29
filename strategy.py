#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with volume spike confirmation and 1d EMA34 trend filter
# Uses proven Camarilla pivot structure (R3/S3 = standard reversal levels) with 1d EMA34 trend filter
# Volume spike (>2.0x 20-period average) confirms institutional participation
# 12h timeframe minimizes fee drag while capturing multi-day moves
# Works in bull/bear: volume confirms breakout validity, 1d EMA34 filters counter-trend noise
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Novelty: Applying proven Camarilla+Volume+Trend framework to 12h timeframe with 1d HTF filter

name = "12h_Camarilla_R3S3_VolumeSpike_1dEMA34_Trend_v1"
timeframe = "12h"
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
    
    # Align daily levels to 12h timeframe (wait for daily bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # 1d EMA34 for trend filter (HTF = 1d as requested)
    df_1d_ema = get_htf_data(prices, '1d')
    if len(df_1d_ema) < 2:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d_ema['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d_ema, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 34)  # warmup for volume MA and 1d EMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(ema_34_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_34 = ema_34_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish entry: price breaks above R3 with volume and above 1d EMA34
                if curr_high > curr_r3 and curr_close > curr_ema_34:
                    signals[i] = 0.30
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below S3 with volume and below 1d EMA34
                elif curr_low < curr_s3 and curr_close < curr_ema_34:
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