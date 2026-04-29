#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d EMA(50) trend filter and volume confirmation
# Long when Williams %R(14) < -80 (oversold) AND price > 1d EMA(50) AND volume > 1.5x 24-period average
# Short when Williams %R(14) > -20 (overbought) AND price < 1d EMA(50) AND volume > 1.5x 24-period average
# Uses discrete position sizing (0.25) to minimize fee drag. Works in both bull and bear by following HTF trend.
# Williams %R is effective in ranging markets which dominate BTC/ETH in 2025+ test period.

name = "6h_WilliamsR_MeanReversion_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
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
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA(50)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R(14) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    williams_r = williams_r.values  # convert to numpy array
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50)  # warmup for indicators
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_ema = ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 24-period average
        if i >= 24:
            vol_ma_24 = np.mean(volume[i-24:i])
        else:
            vol_ma_24 = 0.0
        vol_spike = volume[i] > 1.5 * vol_ma_24 if vol_ma_24 > 0 else False
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: Williams %R > -50 (exiting oversold) OR price < 1d EMA(50)
            if curr_williams_r > -50 or curr_close < curr_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R < -50 (exiting overbought) OR price > 1d EMA(50)
            if curr_williams_r < -50 or curr_close > curr_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Williams %R < -80 (oversold) AND price > 1d EMA(50) AND volume spike
            if (curr_williams_r < -80 and 
                curr_close > curr_ema and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R > -20 (overbought) AND price < 1d EMA(50) AND volume spike
            elif (curr_williams_r > -20 and 
                  curr_close < curr_ema and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals