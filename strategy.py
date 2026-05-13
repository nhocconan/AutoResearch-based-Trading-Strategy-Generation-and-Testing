#!/usr/bin/env python3
# Hypothesis: 6h Williams %R mean reversion with 12h EMA200 trend filter and Bollinger Bands squeeze confirmation.
# Long when Williams %R < -80 (oversold), price > 12h EMA200 (uptrend), and Bollinger Bands width < 20th percentile (low volatility/squeeze).
# Short when Williams %R > -20 (overbought), price < 12h EMA200 (downtrend), and Bollinger Bands width < 20th percentile.
# Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts) indicating mean reversion completion.
# Uses discrete position sizing 0.25. Target: 50-150 total trades over 4 years on 6h timeframe.
# Williams %R captures extreme reversals, 12h EMA200 filters counter-trend trades, Bollinger squeeze identifies low-volatility breakout setups.
# This combination avoids the saturated Camarilla/Donchian families while adding a novel volatility regime filter.

name = "6h_WilliamsR_MeanReversion_12hEMA200_BBSqueeze_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period)
    lookback_wr = 14
    if n < lookback_wr:
        return np.zeros(n)
    
    highest_high = pd.Series(high).rolling(window=lookback_wr, min_periods=lookback_wr).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_wr, min_periods=lookback_wr).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate Bollinger Bands width (20-period, 2 std)
    lookback_bb = 20
    if n < lookback_bb:
        return np.zeros(n)
    
    ma_bb = pd.Series(close).rolling(window=lookback_bb, min_periods=lookback_bb).mean().values
    std_bb = pd.Series(close).rolling(window=lookback_bb, min_periods=lookback_bb).std().values
    upper_bb = ma_bb + 2 * std_bb
    lower_bb = ma_bb - 2 * std_bb
    bb_width = (upper_bb - lower_bb) / ma_bb  # Normalized width
    # Avoid division by zero
    bb_width = np.where(ma_bb == 0, 0, bb_width)
    
    # Calculate 20th percentile of BB width for squeeze condition (using expanding window to avoid look-ahead)
    bb_width_percentile = np.full(n, np.nan)
    for i in range(lookback_bb, n):
        # Use only past data for percentile calculation
        past_widths = bb_width[:i+1]
        valid_widths = past_widths[~np.isnan(past_widths)]
        if len(valid_widths) >= 20:  # Need sufficient data for percentile
            bb_width_percentile[i] = np.percentile(valid_widths, 20)
    
    # Get 12h data for EMA200 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA200 on 12h data
    ema_200_12h = pd.Series(close_12h).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Align 12h EMA200 to 6h timeframe (wait for 12h bar to close)
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start loop after sufficient data for all indicators
    start_idx = max(lookback_wr, lookback_bb, 200) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(bb_width[i]) or 
            np.isnan(bb_width_percentile[i]) or np.isnan(ema_200_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Check Bollinger squeeze condition: width < 20th percentile
        is_squeeze = bb_width[i] < bb_width_percentile[i]
        
        if position == 0:
            # LONG: Williams %R oversold (< -80), uptrend (price > 12h EMA200), and Bollinger squeeze
            if (williams_r[i] < -80 and 
                close[i] > ema_200_12h_aligned[i] and 
                is_squeeze):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R overbought (> -20), downtrend (price < 12h EMA200), and Bollinger squeeze
            elif (williams_r[i] > -20 and 
                  close[i] < ema_200_12h_aligned[i] and 
                  is_squeeze):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses above -50 (mean reversion complete)
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses below -50 (mean reversion complete)
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals