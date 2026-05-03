#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d EMA50 trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) in bull trend (close > 1d EMA50) with volume > 1.8x 20-period MA.
# Short when Williams %R > -20 (overbought) in bear trend (close < 1d EMA50) with volume spike.
# Williams %R captures exhaustion; 1d EMA50 filters countertrend whipsaw; volume confirms participation.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25.

name = "12h_WilliamsR_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - df_1d['close'].values) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Align Williams %R to 12h timeframe (wait for completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume regime: current 12h volume > 1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1d_aligned[i]
        williams_r_val = williams_r_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            if is_bull_trend and williams_r_val < -80 and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and williams_r_val > -20 and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R > -50 (exit oversold) OR trend reversal
            if williams_r_val > -50 or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R < -50 (exit overbought) OR trend reversal
            if williams_r_val < -50 or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals