#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1w Williams %R + 1d EMA trend filter + volume spike confirmation
# - Uses 1w HTF for Williams %R(14) to identify extreme oversold/overbought conditions
# - Uses 1d HTF for EMA(50) trend filter: only long when price > EMA50, short when price < EMA50
# - Entry on 6h timeframe when Williams %R shows reversal from extreme levels with volume confirmation
# - Williams %R long signal: %R crosses above -80 from below (oversold bounce)
# - Williams %R short signal: %R crosses below -20 from above (overbought rejection)
# - Volume confirmation: current 6h volume > 2.0x 24-period average (4 days of 6h bars)
# - Fixed position size 0.25 to control drawdown
# - Works in bull/bear: Williams %R captures mean reversion in ranging markets, EMA filter avoids counter-trend trades
# - Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years)

name = "6h_1w_1d_williams_r_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for Williams %R
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w < 30):
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Williams %R(14)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high_14 - lowest_low_14) != 0,
        ((highest_high_14 - close_1w) / (highest_high_14 - lowest_low_14)) * -100,
        -50.0  # neutral when no range
    )
    
    # Align Williams %R to 6h timeframe (wait for completed 1w bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50)
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Pre-compute volume confirmation (24-period average for 6h = 4 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma_24[i]) or
            vol_ma_24[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 2.0x average
        volume_confirmed = volume[i] > 2.0 * vol_ma_24[i]
        
        if position == 1:  # Long position
            # Exit long: Williams %R crosses below -50 (momentum fading) OR price crosses below EMA50
            if williams_r_aligned[i] < -50.0 or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short: Williams %R crosses above -50 (momentum fading) OR price crosses above EMA50
            if williams_r_aligned[i] > -50.0 or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Williams %R reversal from extreme with volume confirmation and EMA filter
            if volume_confirmed:
                # Long entry: Williams %R crosses above -80 from below AND price > EMA50
                if (williams_r_aligned[i] > -80.0 and 
                    i > 100 and williams_r_aligned[i-1] <= -80.0 and
                    close[i] > ema_50_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short entry: Williams %R crosses below -20 from above AND price < EMA50
                elif (williams_r_aligned[i] < -20.0 and 
                      i > 100 and williams_r_aligned[i-1] >= -20.0 and
                      close[i] < ema_50_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals