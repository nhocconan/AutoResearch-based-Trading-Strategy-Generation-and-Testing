#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R Extreme Reversal with 1d EMA50 trend filter and volume spike confirmation
# Williams %R measures overbought/oversold: > -20 = overbought, < -80 = oversold
# Extreme reversal: Williams %R crosses back above -80 from below (oversold bounce) for long
#                 Williams %R crosses back below -20 from above (overbought rejection) for short
# Trend filter: price above/below 1d EMA50 ensures alignment with higher timeframe trend
# Volume spike: current volume > 2.0 * 20-period average volume confirms momentum
# Uses 12h timeframe for lower frequency (target: 12-37 trades/year) to minimize fee drag
# Works in bull markets via buying oversold dips in uptrend and in bear markets via selling overbought rallies in downtrend

name = "12h_WilliamsR_Extreme_Reversal_1dEMA50_Trend_VolumeSpike_v1"
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
    
    # 1d HTF data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R (14-period): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, 14, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(highest_high_14[i]) or np.isnan(lowest_low_14[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Williams %R extreme reversal conditions
        # Long: Williams %R crosses above -80 from below (oversold bounce)
        williams_r_long = (williams_r[i] > -80) and (williams_r[i-1] <= -80)
        # Short: Williams %R crosses below -20 from above (overbought rejection)
        williams_r_short = (williams_r[i] < -20) and (williams_r[i-1] >= -20)
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Oversold bounce, volume spike, uptrend
            if williams_r_long and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Overbought rejection, volume spike, downtrend
            elif williams_r_short and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Williams %R >= -20 (overbought) or trend reversal
            if williams_r[i] >= -20 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Williams %R <= -80 (oversold) or trend reversal
            if williams_r[i] <= -80 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals