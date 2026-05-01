#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R reversal with 1d EMA50 trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. Long when %R crosses above -80 from below,
# Short when %R crosses below -20 from above. Confirmed by 1d EMA50 trend and volume spike (>1.8x 20-bar MA).
# Works in both bull and bear markets via mean reversion with trend alignment.
# Target: 80-180 total trades over 4 years (20-45/year) with discrete sizing (0.25).

name = "4h_WilliamsR_Reversal_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily HTF data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation: current volume > 1.8 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 20  # Need 20 for volume MA and Williams %R
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_4h[i]) or np.isnan(williams_r[i]) or np.isnan(volume_ma_20[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 from below, above daily EMA50, and volume confirmation
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and curr_close > ema_50_4h[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above, below daily EMA50, and volume confirmation
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and curr_close < ema_50_4h[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Williams %R >= -20 (overbought) or below daily EMA50
            if williams_r[i] >= -20 or curr_close < ema_50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Williams %R <= -80 (oversold) or above daily EMA50
            if williams_r[i] <= -80 or curr_close > ema_50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals