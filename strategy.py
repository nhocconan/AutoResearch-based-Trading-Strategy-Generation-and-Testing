#!/usr/bin/env python3
"""
1h_Bollinger_Squeeze_4hEMA34_Trend
Hypothesis: Trade Bollinger Band squeezes on 1h with 4h EMA34 trend filter. Squeeze occurs when BB width < 50th percentile of last 200 periods. Enter long when price > upper band and 4h EMA34 trending up (current > previous), short when price < lower band and 4h EMA34 trending down. Target 20-40 trades/year via Bollinger squeeze rarity + trend alignment. Works in bull/bear by following 4h trend. Uses volume confirmation > 1.5x 24-period average to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA(34)
    close_4h = df_4h['close'].values
    ema_period = 34
    ema_4h = np.full_like(close_4h, np.nan)
    
    if len(close_4h) >= ema_period:
        ema_4h[ema_period - 1] = np.mean(close_4h[:ema_period])
        for i in range(ema_period, len(close_4h)):
            ema_4h[i] = (close_4h[i] * 2 / (ema_period + 1)) + (ema_4h[i-1] * (ema_period - 1) / (ema_period + 1))
    
    # Align 4h EMA to 1h timeframe
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = np.full_like(close, np.nan)
    std_dev = np.full_like(close, np.nan)
    upper_band = np.full_like(close, np.nan)
    lower_band = np.full_like(close, np.nan)
    bb_width = np.full_like(close, np.nan)
    
    for i in range(bb_period, n):
        sma[i] = np.mean(close[i - bb_period:i])
        std_dev[i] = np.std(close[i - bb_period:i])
        upper_band[i] = sma[i] + bb_std * std_dev[i]
        lower_band[i] = sma[i] - bb_std * std_dev[i]
        bb_width[i] = upper_band[i] - lower_band[i]
    
    # Bollinger Band width percentile (50th percentile of last 200)
    width_percentile = np.full_like(bb_width, np.nan)
    lookback = 200
    
    for i in range(lookback, n):
        width_window = bb_width[i - lookback:i]
        width_window = width_window[~np.isnan(width_window)]
        if len(width_window) > 0:
            width_percentile[i] = np.percentile(width_window, 50)
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, vol_period, ema_period, lookback)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(width_percentile[i])):
            signals[i] = 0.0
            continue
        
        # Bollinger squeeze condition: width < 50th percentile
        squeeze = bb_width[i] < width_percentile[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price > upper band + squeeze + volume + 4h EMA trending up
            if (close[i] > upper_band[i] and squeeze and vol_confirm and 
                i > 0 and not np.isnan(ema_4h_aligned[i-1]) and ema_4h_aligned[i] > ema_4h_aligned[i-1]):
                signals[i] = 0.20
                position = 1
            # Short: price < lower band + squeeze + volume + 4h EMA trending down
            elif (close[i] < lower_band[i] and squeeze and vol_confirm and 
                  i > 0 and not np.isnan(ema_4h_aligned[i-1]) and ema_4h_aligned[i] < ema_4h_aligned[i-1]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price < lower band or 4h EMA turns down
            if close[i] < lower_band[i] or (i > 0 and not np.isnan(ema_4h_aligned[i-1]) and ema_4h_aligned[i] < ema_4h_aligned[i-1]):
                signals[i] = -0.20  # reverse to short
                position = -1
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price > upper band or 4h EMA turns up
            if close[i] > upper_band[i] or (i > 0 and not np.isnan(ema_4h_aligned[i-1]) and ema_4h_aligned[i] > ema_4h_aligned[i-1]):
                signals[i] = 0.20  # reverse to long
                position = 1
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Bollinger_Squeeze_4hEMA34_Trend"
timeframe = "1h"
leverage = 1.0