#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian channel breakout with 1-day EMA trend filter and volume confirmation.
# Buy when price breaks above 20-period Donchian high in uptrend (price > 1-day EMA50) with volume > 1.5x average.
# Sell when price breaks below 20-period Donchian low in downtrend (price < 1-day EMA50) with volume > 1.5x average.
# Designed for 4h timeframe to target 75-200 trades over 4 years. Uses tight entry conditions to minimize fee drag.
# Works in bull/bear markets via EMA-based directional bias and breakout entries aligned with trend.

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
    
    # Donchian channel (20-period) on 4h data
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
        
        # Trend bias: bullish if price above 1-day EMA50, bearish if below
        bullish_bias = close[i] > ema_50_aligned[i]
        bearish_bias = close[i] < ema_50_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i-1]  # break above previous period's high
        breakout_down = close[i] < donchian_low[i-1]  # break below previous period's low
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian low or stoploss (2x ATR approximation)
            atr_approx = (high[i] - low[i])  # simple range approximation
            if atr_approx > 0:
                stop_loss_level = entry_price - 2.0 * atr_approx
            else:
                stop_loss_level = entry_price - 2.0 * 0.001
            
            if (breakout_down or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high or stoploss
            atr_approx = (high[i] - low[i])
            if atr_approx > 0:
                stop_loss_level = entry_price + 2.0 * atr_approx
            else:
                stop_loss_level = entry_price + 2.0 * 0.001
            
            if (breakout_up or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in direction of EMA trend with volume confirmation
            if volume_filter:
                # Long: breakout above Donchian high in uptrend
                if breakout_up and bullish_bias:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakout below Donchian low in downtrend
                elif breakout_down and bearish_bias:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals