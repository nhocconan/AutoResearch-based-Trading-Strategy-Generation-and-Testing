#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1w EMA50 trend filter and volume confirmation.
# Long when Williams %R crosses above -80 from below, price > 1w EMA50, and volume > 1.5x 24-bar avg.
# Short when Williams %R crosses below -20 from above, price < 1w EMA50, and volume > 1.5x 24-bar avg.
# Exit when Williams %R crosses -50 (mean reversion to midline).
# Williams %R identifies overbought/oversold conditions, effective in ranging markets.
# Combined with 1w EMA50 trend filter to avoid counter-trend trades and volume confirmation to reduce false signals.
# Timeframe: 12h as per experiment guidelines.

name = "12h_WilliamsR_MeanRev_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams %R (14-period) using 12h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: volume > 1.5x 24-period average (24*12h = 12d)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (1.5 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and Williams %R
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_williams_r = williams_r[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 from below, price > 1w EMA50, volume spike
            if (curr_williams_r > -80 and 
                williams_r[i-1] <= -80 and  # crossed above -80
                curr_close > curr_ema_50_1w and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above, price < 1w EMA50, volume spike
            elif (curr_williams_r < -20 and 
                  williams_r[i-1] >= -20 and  # crossed below -20
                  curr_close < curr_ema_50_1w and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: Williams %R crosses above -50 (mean reversion to midline)
            if curr_williams_r >= -50 and williams_r[i-1] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: Williams %R crosses below -50 (mean reversion to midline)
            if curr_williams_r <= -50 and williams_r[i-1] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals