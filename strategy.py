#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R Mean Reversion with 1d EMA50 trend filter and volume spike confirmation
# Williams %R identifies overbought/oversold conditions: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
# Long when %R < -80 (oversold) AND price > 1d EMA50 (uptrend bias) AND volume > 2.0x 20-period average
# Short when %R > -20 (overbought) AND price < 1d EMA50 (downtrend bias) AND volume > 2.0x 20-period average
# Exit when %R returns to neutral range (-50 to -50) or opposite signal
# Designed for ~20-40 trades/year on 4h timeframe to minimize fee drag
# Uses 1d trend filter to avoid counter-trend trades in strong markets
# Volume spike filter reduces false signals and ensures participation

name = "4h_WilliamsR_MeanRev_1dEMA50_VolumeSpike_v1"
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
    
    # Get 1d data for EMA50 trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R (14-period) on 4h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate 20-period average volume for spike confirmation (on 4h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # Volume MA and Williams %R warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Williams %R returns to neutral (> -50) or opposite signal
            if curr_williams_r > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R returns to neutral (< -50) or opposite signal
            if curr_williams_r < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume spike confirmation: current volume > 2.0x 20-period average
            vol_spike = curr_volume > 2.0 * curr_vol_ma
            
            # Long entry: Williams %R oversold (< -80) in uptrend (price > 1d EMA50)
            if vol_spike and curr_close > curr_ema50_1d:
                if curr_williams_r < -80:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
            # Short entry: Williams %R overbought (> -20) in downtrend (price < 1d EMA50)
            elif vol_spike and curr_close < curr_ema50_1d:
                if curr_williams_r > -20:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals