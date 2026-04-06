#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian(20) breakout with 1-week EMA40 trend filter and volume confirmation.
# Designed for 12h timeframe targeting 50-150 total trades over 4 years.
# Donchian breakouts capture momentum bursts. EMA40 on weekly provides strong trend bias.
# Volume confirmation ensures institutional participation. Works in bull/bear via EMA filter.
# Uses proper ATR-based stoploss to limit drawdown.

name = "12h_donchian20_1w_ema40_vol_v1"
timeframe = "12h"
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
    
    # 1-week EMA40 for trend bias
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA40 on 1w closes
    ema_40_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 40:
        ema_40_1w[39] = np.mean(close_1w[:40])
        for i in range(40, len(close_1w)):
            ema_40_1w[i] = (close_1w[i] * 2 / 41) + (ema_40_1w[i-1] * 39 / 41)
    
    # Align EMA40 to 12h timeframe (shifted by 1 1w bar for no look-ahead)
    ema_40_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    # Donchian Channel (20-period) on 12h data
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    # ATR(14) for stoploss
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first bar
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(40, n):  # Start after EMA warmup
        # Skip if required data not available
        if (np.isnan(ema_40_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x 20-period average
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Trend bias: long above EMA40, short below EMA40
        bullish_bias = close[i] > ema_40_aligned[i]
        bearish_bias = close[i] < ema_40_aligned[i]
        
        # Donchian breakout conditions
        breakout_high = close[i] > donchian_high[i-1]  # break above previous high
        breakout_low = close[i] < donchian_low[i-1]    # break below previous low
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Donchian break below or stoploss (2x ATR)
            if i > 0:
                stop_loss_level = entry_price - 2.0 * atr[i-1]
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
            if i > 0:
                stop_loss_level = entry_price + 2.0 * atr[i-1]
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