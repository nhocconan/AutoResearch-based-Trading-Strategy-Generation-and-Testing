#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike (>2x 20-bar MA)
# Camarilla R3/S3 levels act as magnet/pivot points; breakouts with volume and trend alignment capture strong moves.
# Works in bull markets via long breakouts above R3 with uptrend, in bear markets via short breakdowns below S3 with downtrend.
# Volume spike confirms institutional participation. Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.25).

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
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
    
    # 1d HTF data for Camarilla pivot and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Prior day's OHLC for Camarilla calculation (standard formula)
    # Camarilla levels use prior day's range to calculate support/resistance
    prior_high = df_1d['high'].shift(1).values  # prior day's high
    prior_low = df_1d['low'].shift(1).values    # prior day's low
    prior_close = df_1d['close'].shift(1).values # prior day's close
    
    # Calculate Camarilla levels
    # R4 = prior_close + ((prior_high - prior_low) * 1.1/2)
    # R3 = prior_close + ((prior_high - prior_low) * 1.1/4)
    # S3 = prior_close - ((prior_high - prior_low) * 1.1/4)
    # S4 = prior_close - ((prior_high - prior_low) * 1.1/2)
    # We only need R3 and S3 for this strategy
    camarilla_range = prior_high - prior_low
    r3 = prior_close + (camarilla_range * 1.1 / 4.0)
    s3 = prior_close - (camarilla_range * 1.1 / 4.0)
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(34, 20)  # Need 34 for EMA and 20 for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(volume_ma_20[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3, above EMA34 trend, and volume spike
            if curr_close > r3_aligned[i] and curr_close > ema_34_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, below EMA34 trend, and volume spike
            elif curr_close < s3_aligned[i] and curr_close < ema_34_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price breaking below S3 or below EMA34
            if curr_close < s3_aligned[i] or curr_close < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on price breaking above R3 or above EMA34
            if curr_close > r3_aligned[i] or curr_close > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals