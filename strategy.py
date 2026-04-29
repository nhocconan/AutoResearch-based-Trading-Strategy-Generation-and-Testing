#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based stoploss
# Long when price breaks above Donchian upper (20-bar high) AND volume > 1.5x 20-bar avg
# Short when price breaks below Donchian lower (20-bar low) AND volume > 1.5x 20-bar avg
# Exit when price retraces 50% of the breakout move or hits ATR stop (2x ATR)
# Uses discrete position sizing (0.25) to reduce fee drag
# Designed for 4h timeframe to target 20-50 trades/year, avoiding overtrading
# Works in bull markets by capturing breakouts and in bear markets by shorting breakdowns
# Volume confirmation filters out low-momentum false breakouts
# ATR stoploss manages risk during adverse moves

name = "4h_Donchian20_VolumeConfirm_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels: 20-bar high/low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = 20  # Donchian and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_close = close[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_atr = atr[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit 1: price retraces 50% of breakout move
            retrace_level = entry_price + 0.5 * (curr_high - entry_price)  # approximate using current high
            if curr_close <= retrace_level:
                signals[i] = 0.0
                position = 0
            # Exit 2: ATR stoploss (2x ATR below entry)
            elif curr_close < entry_price - 2.0 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit 1: price retraces 50% of breakout move
            retrace_level = entry_price - 0.5 * (entry_price - curr_low)  # approximate using current low
            if curr_close >= retrace_level:
                signals[i] = 0.0
                position = 0
            # Exit 2: ATR stoploss (2x ATR above entry)
            elif curr_close > entry_price + 2.0 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian upper AND volume confirmation
            if curr_close > curr_donchian_high and vol_conf:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_at_entry = curr_atr
            # Short when price breaks below Donchian lower AND volume confirmation
            elif curr_close < curr_donchian_low and vol_conf:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals