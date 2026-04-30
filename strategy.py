#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Mean Reversion with 1w EMA50 trend filter and volume confirmation.
# Williams %R measures overbought/oversold: values below -80 = oversold, above -20 = overbought.
# Long when Williams %R < -80 (oversold) AND price > 1w EMA50 (uptrend) AND volume > 2.0x 20-bar average.
# Short when Williams %R > -20 (overbought) AND price < 1w EMA50 (downtrend) AND volume > 2.0x 20-bar average.
# Exit when Williams %R reverts to midpoint (-50) or opposite extreme is reached.
# Williams %R is effective in ranging markets and catches reversals in bear markets.
# 1w EMA50 filters for dominant long-term trend to avoid counter-trend entries.
# Volume confirmation (2.0x) ensures institutional participation and reduces false signals.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

name = "6h_WilliamsR_MeanRev_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams %R calculation (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(35, 20)  # warmup for Williams %R and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_wr = williams_r[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Oversold (Williams %R < -80), uptrend (price > 1w EMA50), volume confirmation
            if (curr_wr < -80 and 
                curr_close > ema_50_1w_aligned[i] and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Overbought (Williams %R > -20), downtrend (price < 1w EMA50), volume confirmation
            elif (curr_wr > -20 and 
                  curr_close < ema_50_1w_aligned[i] and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: Williams %R reverts to midpoint (-50) or reaches overbought (-20)
            if curr_wr >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: Williams %R reverts to midpoint (-50) or reaches oversold (-80)
            if curr_wr <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals