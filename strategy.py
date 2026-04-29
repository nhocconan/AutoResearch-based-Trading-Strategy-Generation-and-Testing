#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1w EMA50 trend filter and volume confirmation
# Long when Williams %R crosses above -80 from oversold, price > 1w EMA50, and volume spike
# Short when Williams %R crosses below -20 from overbought, price < 1w EMA50, and volume spike
# Uses mean reversion in extremes with HTF trend alignment and volume confirmation
# Target: 50-150 total trades over 4 years (12-37/year) for optimal fee drag balance

name = "6h_WilliamsR_VolumeSpike_1wEMA50_Trend_v1"
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
    
    # Calculate Williams %R (14-period)
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 1-week EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 20)  # warmup for Williams %R and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_wr = williams_r[i]
        curr_ema_50 = ema_50_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish entry: Williams %R crosses above -80 from below (oversold reversal)
                if i > start_idx and williams_r[i-1] <= -80 and curr_wr > -80 and curr_close > curr_ema_50:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Williams %R crosses below -20 from above (overbought reversal)
                elif i > start_idx and williams_r[i-1] >= -20 and curr_wr < -20 and curr_close < curr_ema_50:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when Williams %R crosses above -20 (overbought) or below -80 (stop)
            if curr_wr >= -20 or curr_wr <= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Williams %R crosses below -80 (oversold) or above -20 (stop)
            if curr_wr <= -80 or curr_wr >= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals