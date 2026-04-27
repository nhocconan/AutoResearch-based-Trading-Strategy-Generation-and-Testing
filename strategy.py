#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d EMA34 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions for mean reversion entries.
# Long when Williams %R < -80 (oversold) and price above 1d EMA34 (uptrend).
# Short when Williams %R > -20 (overbought) and price below 1d EMA34 (downtrend).
# Volume > 1.5x 20-period average confirms momentum.
# Exit when Williams %R returns to -50 (neutral) or trend reverses.
# Target: 15-25 trades/year to minimize fee decay while capturing high-probability reversals.
# Focus on BTC/ETH as primary assets with proven mean-reversion edge in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Williams %R (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    williams_r = np.full(len(close_1d), np.nan)
    williams_period = 14
    for i in range(williams_period, len(close_1d)):
        highest_high = np.max(high_1d[i-williams_period:i+1])
        lowest_low = np.min(low_1d[i-williams_period:i+1])
        if highest_high - lowest_low != 0:
            williams_r[i] = -100 * (highest_high - close_1d[i]) / (highest_high - lowest_low)
        else:
            williams_r[i] = -50  # neutral when no range
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Williams %R and EMA34 to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 20-period average volume for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(williams_period, vol_period, 1)
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        williams_r_val = williams_r_aligned[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Determine trend from 1d EMA34
        uptrend = price > ema_34_1d_aligned[i]
        downtrend = price < ema_34_1d_aligned[i]
        
        # Williams %R conditions
        oversold = williams_r_val < -80
        overbought = williams_r_val > -20
        neutral = -50 <= williams_r_val <= -50  # exit at -50
        
        # Volume confirmation: spike > 1.5x average
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long: oversold with uptrend and volume
            if oversold and uptrend and volume_confirmation:
                signals[i] = size
                position = 1
            # Short: overbought with downtrend and volume
            elif overbought and downtrend and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: returns to neutral or trend breaks down
            if williams_r_val >= -50 or price < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: returns to neutral or trend breaks up
            if williams_r_val <= -50 or price > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_WilliamsR_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0