#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian breakout with 1-week EMA trend filter and volume confirmation.
# Donchian(20) breakout captures momentum in trending markets.
# EMA20 on 1-week provides trend bias: only long when price > EMA20, short when price < EMA20.
# Volume confirmation (current volume > 1.5x 20-period average) ensures institutional participation.
# Designed for 1d timeframe to target 30-100 trades over 4 years.
# Works in bull/bear markets via EMA-based directional bias and breakout entries.

name = "1d_donchian20_1w_ema20_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-week EMA20 for trend bias
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA20 on 1w closes
    ema_20_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema_20_1w[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = (close_1w[i] * 2 / 21) + (ema_20_1w[i-1] * 19 / 21)
    
    # Align EMA20 to 1d timeframe (shifted by 1 1w bar for no look-ahead)
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Donchian Channel (20-period) on 1d data
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_20_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x 20-period average
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Trend bias: long above EMA20, short below EMA20
        bullish_bias = close[i] > ema_20_aligned[i]
        bearish_bias = close[i] < ema_20_aligned[i]
        
        # Donchian breakout conditions
        breakout_high = close[i] > donchian_high[i-1]  # break above previous high
        breakout_low = close[i] < donchian_low[i-1]    # break below previous low
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Donchian break below or stoploss (2x ATR approximation)
            atr_approx = (high[i] - low[i])  # simple range approximation
            if atr_approx > 0:
                stop_loss_level = entry_price - 2.0 * atr_approx
            else:
                stop_loss_level = entry_price - 2.0 * 0.001
            
            if (breakout_low or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Donchian break above or stoploss
            atr_approx = (high[i] - low[i])
            if atr_approx > 0:
                stop_loss_level = entry_price + 2.0 * atr_approx
            else:
                stop_loss_level = entry_price + 2.0 * 0.001
            
            if (breakout_high or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in direction of EMA trend with volume confirmation
            if volume_filter:
                # Long: breakout above Donchian high in uptrend
                if breakout_high and bullish_bias:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakout below Donchian low in downtrend
                elif breakout_low and bearish_bias:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals