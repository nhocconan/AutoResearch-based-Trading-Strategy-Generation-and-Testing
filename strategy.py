#!/usr/bin/env python3
"""
Hypothesis: 1d Williams %R mean reversion with 1w trend filter and volume confirmation.
Long when Williams %R < -80 (oversold) AND price > 1w EMA34 (bullish trend) AND 1d volume > 1.5x 20-bar average volume.
Short when Williams %R > -20 (overbought) AND price < 1w EMA34 (bearish trend) AND 1d volume > 1.5x 20-bar average volume.
Exit when Williams %R crosses above -50 (for long) or below -50 (for short).
Uses 1d for Williams %R and volume, 1w for trend filter. Designed to capture reversals in both bull and bear markets.
Target: 10-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low + 1e-10) * -100
    
    # Calculate 1d volume MA for confirmation
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all 1d indicators to 1d timeframe (no alignment needed as we're using 1d timeframe)
    # But we need to align 1w EMA to 1d timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i]) or
            np.isnan(ema34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-bar average
        volume_confirmed = volume_1d[i] > 1.5 * vol_ma_20[i]
        
        # Williams %R conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        exit_long = williams_r[i] > -50
        exit_short = williams_r[i] < -50
        
        # Trend filter: price vs 1w EMA34
        # Note: we're on 1d timeframe, so we use daily close vs aligned 1w EMA
        price_above_ema = close[i] > ema34_1w_aligned[i]
        price_below_ema = close[i] < ema34_1w_aligned[i]
        
        if position == 0:
            # Long: oversold + bullish trend + volume confirmation
            if (oversold and price_above_ema and volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: overbought + bearish trend + volume confirmation
            elif (overbought and price_below_ema and volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses above -50
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses below -50
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsR_MeanReversion_1wEMA34_Volume"
timeframe = "1d"
leverage = 1.0