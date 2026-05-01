#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 12h EMA trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. In strong trends (12h EMA slope),
# extreme %R readings often precede continuations rather than reversals.
# Long: %R < -80 (oversold) AND price > 12h EMA(34) AND volume > 1.5x 20-bar average.
# Short: %R > -20 (overbought) AND price < 12h EMA(34) AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25. Target: 80-180 total trades over 4 years (20-45/year).
# Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend).

name = "6h_WilliamsR_EMA34_Trend_VolumeConfirm_v1"
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
    
    # Load 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:  # Need enough for EMA calculation
        return np.zeros(n)
    
    # 12h EMA(34) calculation
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Williams %R calculation (14-period) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: current 6h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Williams %R and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(williams_r[i]) or np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)  # Volume spike threshold
        
        # Williams %R conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        # Trend filter: price relative to 12h EMA
        uptrend = curr_close > ema_12h_aligned[i]
        downtrend = curr_close < ema_12h_aligned[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: oversold AND uptrend AND volume confirmation
            if (oversold and 
                uptrend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: overbought AND downtrend AND volume confirmation
            elif (overbought and 
                  downtrend and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: overbought condition OR price crosses below 12h EMA
            if (overbought or 
                curr_close < ema_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: oversold condition OR price crosses above 12h EMA
            if (oversold or 
                curr_close > ema_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals