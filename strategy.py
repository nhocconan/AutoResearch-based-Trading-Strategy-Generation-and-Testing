#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume spike confirmation
# - Williams %R(14) identifies overbought/oversold conditions on 6h timeframe
# - 1d EMA(50) provides trend filter: only take longs above EMA, shorts below EMA
# - Volume spike (>2.0x 20-period average) confirms momentum behind the reversal
# - Works in both bull and bear markets by fading extremes in the direction of higher timeframe trend
# - Discrete position sizing ±0.25 to manage drawdown and minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits

name = "6h_1d_williamsr_meanrev_volume_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return signals
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 1d volume average for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute Williams %R on 6h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    diff = highest_high - lowest_low
    williams_r = np.where(diff != 0, -100 * (highest_high - close) / diff, -50)
    
    for i in range(60, n):  # Start after 60-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or np.isnan(volume_sma_20_aligned[i]) or 
            np.isnan(williams_r[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Williams %R levels
        wr = williams_r[i]
        
        # Trend filter: 1d EMA(50)
        above_trend = close_price > ema_50_aligned[i]
        below_trend = close_price < ema_50_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_spike = volume_current > 2.0 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Williams %R oversold (< -80) + above 1d EMA trend + volume spike
        if wr < -80 and above_trend and vol_spike:
            enter_long = True
        
        # Short: Williams %R overbought (> -20) + below 1d EMA trend + volume spike
        if wr > -20 and below_trend and vol_spike:
            enter_short = True
        
        # Exit conditions: Williams %R reverts to mean (-50) or opposite extreme
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long when Williams %R reverts to mean or goes overbought
            exit_long = (wr >= -50) or (wr > -20)
        elif position == -1:
            # Exit short when Williams %R reverts to mean or goes oversold
            exit_short = (wr <= -50) or (wr < -80)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals