#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversals with 1d EMA34 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions. Extreme readings (> -10 for oversold, < -90 for overbought)
# combined with 1d EMA trend filter captures mean reversals in the direction of higher timeframe trend.
# Volume confirmation ensures reversals have institutional participation. Works in bull (buy dips in uptrend)
# and bear (sell rallies in downtrend). Target: 12-37 trades/year to minimize fee drag.

name = "6h_WilliamsR14_Extreme_1dEMA34_Trend_VolumeConfirm_v1"
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
    
    # 1d HTF data for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams %R(14) on 6h
    highest_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_14 - close) / (highest_14 - lowest_14)
    
    # Volume confirmation: current volume > 1.3 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 34  # Need sufficient history for Williams %R and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(highest_14[i]) or np.isnan(lowest_14[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_wr = williams_r[i]
        
        # Trend filter: price above/below 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Williams %R extreme conditions
        oversold = curr_wr > -10   # Extreme oversold
        overbought = curr_wr < -90  # Extreme overbought
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: extreme oversold, volume spike, uptrend
            if oversold and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: extreme overbought, volume spike, downtrend
            elif overbought and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit when Williams %R returns to neutral or trend reverses
            if curr_wr < -50 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Williams %R returns to neutral or trend reverses
            if curr_wr > -50 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals