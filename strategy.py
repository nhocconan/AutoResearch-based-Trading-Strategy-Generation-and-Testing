#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d EMA34 trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold), 1d EMA34 uptrend (close > EMA34), and volume > 2.0x 20-bar avg.
# Short when Williams %R > -20 (overbought), 1d EMA34 downtrend (close < EMA34), and volume > 2.0x 20-bar avg.
# Exit on opposite signal (Williams %R > -50 for long exit, Williams %R < -50 for short exit).
# Williams %R identifies extreme price levels for mean reversion. Combined with 1d EMA34 trend filter
# and volume confirmation to reduce false signals and capture BTC/ETH moves in both bull and bear markets.
# Timeframe: 4h, HTF: 1d as per experiment guidelines.

name = "4h_WilliamsR_MeanRev_1dEMA34_Trend_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R (14-period) on 4h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for Williams %R and EMA34
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_williams_r = williams_r[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_close = close[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R < -80 (oversold), uptrend (close > 1d EMA34), volume spike
            if (curr_williams_r < -80 and 
                curr_close > curr_ema_34_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought), downtrend (close < 1d EMA34), volume spike
            elif (curr_williams_r > -20 and 
                  curr_close < curr_ema_34_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: Williams %R > -50 (exiting oversold territory)
            if curr_williams_r > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: Williams %R < -50 (exiting overbought territory)
            if curr_williams_r < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals