#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter and volume confirmation.
# Uses Donchian channel breakouts from 4h data, filtered by 12h EMA trend direction.
# Volume confirmation (current volume > 1.3x 20-period average) filters low-quality breakouts.
# Designed for 4h timeframe to target 75-200 trades over 4 years (19-50/year).
# Works in bull/bear markets via 12h EMA trend bias - only trades in direction of higher timeframe trend.

name = "4h_donchian20_12h_ema_vol_v1"
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
    
    # 12h EMA50 for trend bias
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 on 12h closes
    ema_50_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema_50_12h[49] = np.mean(close_12h[:50])
        for i in range(50, len(close_12h)):
            ema_50_12h[i] = (close_12h[i] * 2 / 51) + (ema_50_12h[i-1] * 49 / 51)
    
    # Align EMA50 to 4h timeframe (shifted by 1 12h bar for no look-ahead)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian Channel (20-period) on 4h data
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    for i in range(19, n):
        upper_channel[i] = np.max(high[i-19:i+1])
        lower_channel[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Start after Donchian channel is available
        # Skip if required data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.3x 20-period average
        vol_ma = np.mean(volume[max(0, i-19):i+1]) if i >= 19 else np.nan
        if np.isnan(vol_ma):
            volume_filter = False
        else:
            volume_filter = volume[i] > vol_ma * 1.3
        
        # Trend bias: 12h EMA50
        bullish_bias = close[i] > ema_50_aligned[i]
        bearish_bias = close[i] < ema_50_aligned[i]
        
        # Donchian breakout conditions
        breakout_above = close[i] > upper_channel[i] and close[i-1] <= upper_channel[i-1]
        breakout_below = close[i] < lower_channel[i] and close[i-1] >= lower_channel[i-1]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: re-entry into channel or stoploss (2x ATR approximation)
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.0 * atr_approx
            
            if (close[i] < upper_channel[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: re-entry into channel or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.0 * atr_approx
            
            if (close[i] > lower_channel[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in direction of 12h trend with volume confirmation
            if volume_filter:
                # Long: breakout above upper channel in uptrend
                if breakout_above and bullish_bias:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakout below lower channel in downtrend
                elif breakout_below and bearish_bias:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals