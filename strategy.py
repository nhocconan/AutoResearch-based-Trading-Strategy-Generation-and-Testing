#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R mean reversion with 1w trend filter and volume confirmation
# Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
# Oversold: %R < -80, Overbought: %R > -20
# Long when Williams %R crosses above -80 from below AND price > 1w EMA50 AND volume > 1.5x 20-period average
# Short when Williams %R crosses below -20 from above AND price < 1w EMA50 AND volume > 1.5x 20-period average
# Uses 1d timeframe for lower trade frequency (~10-25 trades/year) and discrete sizing (0.25) to minimize fee drag
# Works in both bull and bear markets by combining mean reversion (Williams %R) with trend filter (1w EMA50)

name = "1d_WilliamsR_MeanReversion_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50, 14)  # warmup for indicators
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_williams_r_prev = williams_r[i-1]
        curr_ema_1w = ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 1.5 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits
        if position == 1:  # Long position
            # Exit conditions: Williams %R crosses below -50 OR price crosses below 1w EMA50
            if curr_williams_r < -50 or curr_close < curr_ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: Williams %R crosses above -50 OR price crosses above 1w EMA50
            if curr_williams_r > -50 or curr_close > curr_ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Williams %R crosses above -80 from below AND price > 1w EMA50 AND volume spike
            if (curr_williams_r > -80 and 
                curr_williams_r_prev <= -80 and  # Cross above -80
                curr_close > curr_ema_1w and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R crosses below -20 from above AND price < 1w EMA50 AND volume spike
            elif (curr_williams_r < -20 and 
                  curr_williams_r_prev >= -20 and  # Cross below -20
                  curr_close < curr_ema_1w and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals