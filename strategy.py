#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 12h EMA50 trend filter + volume confirmation.
# Williams %R identifies overbought/oversold conditions. Long when %R crosses above -80 from below,
# Short when %R crosses below -20 from above. Confirmed by 12h EMA50 trend and volume spike (>1.5x 24-bar MA).
# Works in ranging markets with mean reversion and in trending markets with pullback entries.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.25).

name = "6h_WilliamsR_Extreme_12hEMA50_Trend_VolumeConfirm_v1"
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
    
    # 12h HTF data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams %R (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Volume confirmation: current volume > 1.5 * 24-period average volume
    volume_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (volume_ma_24 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 24  # Need 24 for volume MA, 14 for Williams %R
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_6h[i]) or np.isnan(williams_r[i]) or np.isnan(highest_high_14[i]) or np.isnan(lowest_low_14[i]) or np.isnan(volume_ma_24[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 from below, above 12h EMA50, and volume confirmation
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and curr_close > ema_50_6h[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above, below 12h EMA50, and volume confirmation
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and curr_close < ema_50_6h[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Williams %R >= -20 (overbought) or below 12h EMA50
            if williams_r[i] >= -20 or curr_close < ema_50_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Williams %R <= -80 (oversold) or above 12h EMA50
            if williams_r[i] <= -80 or curr_close > ema_50_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals