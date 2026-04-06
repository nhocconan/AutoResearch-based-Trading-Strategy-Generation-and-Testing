#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 1-day EMA trend filter and volume confirmation.
# Only enter long when price breaks above 4h Donchian upper band AND price > 1d EMA50 (uptrend).
# Only enter short when price breaks below 4h Donchian lower band AND price < 1d EMA50 (downtrend).
# Volume > 1.5x 20-period average confirms institutional participation.
# Uses ATR-based stoploss (2x ATR) to limit drawdown. Designed for 4h timeframe to target 75-200 trades over 4 years.
# Works in bull/bear markets via EMA-based directional bias and breakout entries with volume confirmation.

name = "4h_donchian20_1d_ema50_vol_v1"
timeframe = "4h"
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
    
    # 1-day EMA50 for trend bias
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d closes
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 / 51) + (ema_50_1d[i-1] * 49 / 51)
    
    # Align EMA50 to 4h timeframe (shifted by 1 1d bar for no look-ahead)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4-hour Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Volume confirmation: 4h volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x 20-period average
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Trend bias: long above EMA50, short below EMA50
        bullish_bias = close[i] > ema_50_aligned[i]
        bearish_bias = close[i] < ema_50_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i-1]  # break above previous upper band
        breakout_down = close[i] < donchian_low[i-1]  # break below previous lower band
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Calculate ATR approximation for stoploss
            atr_approx = np.max([high[i] - low[i], 
                                abs(high[i] - close[i-1]), 
                                abs(low[i] - close[i-1])])
            if atr_approx > 0:
                stop_loss_level = entry_price - 2.0 * atr_approx
            else:
                stop_loss_level = entry_price - 2.0 * 0.001
            
            # Exit: price breaks below Donchian lower band OR stoploss hit
            if (close[i] < donchian_low[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Calculate ATR approximation for stoploss
            atr_approx = np.max([high[i] - low[i], 
                                abs(high[i] - close[i-1]), 
                                abs(low[i] - close[i-1])])
            if atr_approx > 0:
                stop_loss_level = entry_price + 2.0 * atr_approx
            else:
                stop_loss_level = entry_price + 2.0 * 0.001
            
            # Exit: price breaks above Donchian upper band OR stoploss hit
            if (close[i] > donchian_high[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in direction of EMA trend with volume confirmation
            if volume_filter:
                # Long: breakout above Donchian upper band in uptrend
                if breakout_up and bullish_bias:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakout below Donchian lower band in downtrend
                elif breakout_down and bearish_bias:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals