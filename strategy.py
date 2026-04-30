#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R mean reversion with 1w EMA50 trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions: Long when %R crosses above -80 from below (oversold bounce),
# Short when %R crosses below -20 from above (overbought reversal). 1w EMA50 ensures alignment with dominant long-term
# trend to avoid counter-trend entries in bear markets. Volume confirmation (>1.5x 20-bar average) filters weak moves.
# Exit on %R crossing -50 (mean reversion midpoint) or opposite signal. Designed for low trade frequency (~15-25/year)
# to minimize fee drag while capturing swing reversals in both bull and bear regimes.

name = "1d_WilliamsR_MeanRev_1wEMA50_Trend_VolumeConfirmation_v1"
timeframe = "1d"
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
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Williams %R and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_volume_confirm = volume_confirm[i]
        prev_williams_r = williams_r[i-1]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 from below (oversold bounce), uptrend (price > 1w EMA50), volume confirmation
            if (curr_williams_r > -80 and prev_williams_r <= -80 and 
                curr_close > ema_50_1w_aligned[i] and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above (overbought reversal), downtrend (price < 1w EMA50), volume confirmation
            elif (curr_williams_r < -20 and prev_williams_r >= -20 and 
                  curr_close < ema_50_1w_aligned[i] and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses above -50 (mean reversion midpoint) or opposite short signal
            if (curr_williams_r > -50 and prev_williams_r <= -50) or \
               (curr_williams_r < -20 and prev_williams_r >= -20 and curr_close < ema_50_1w_aligned[i] and curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -50 (mean reversion midpoint) or opposite long signal
            if (curr_williams_r < -50 and prev_williams_r >= -50) or \
               (curr_williams_r > -80 and prev_williams_r <= -80 and curr_close > ema_50_1w_aligned[i] and curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals