#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based stoploss.
# Long when price breaks above upper Donchian channel AND volume > 2.0x 20-bar avg.
# Short when price breaks below lower Donchian channel AND volume > 2.0x 20-bar avg.
# Exit on opposite Donchian channel touch or ATR(14) stoploss (2.5x ATR).
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 20-50 trades/year on 4h.
# Works in bull via breakout continuation, works in bear via volume spike requirement capturing panic climaxes.

name = "4h_Donchian20_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_channel = high_series.rolling(window=20, min_periods=20).max().values
    lower_channel = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >2.0x 20-bar average volume (strict filter to reduce trades)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = 20  # Need sufficient history for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        upper = upper_channel[i]
        lower = lower_channel[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Handle position management and exits
        if position == 1:  # Long position
            # Check stoploss: price < entry_price - 2.5 * atr_at_entry
            if curr_low <= entry_price - 2.5 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            # Check exit: price touches lower Donchian channel
            elif curr_low <= lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Check stoploss: price > entry_price + 2.5 * atr_at_entry
            if curr_high >= entry_price + 2.5 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            # Check exit: price touches upper Donchian channel
            elif curr_high >= upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:  # Flat - look for new entries
            # Long when price breaks above upper Donchian AND volume confirmation
            if curr_close > upper and vol_conf:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_at_entry = atr[i]
            # Short when price breaks below lower Donchian AND volume confirmation
            elif curr_close < lower and vol_conf:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_at_entry = atr[i]
            else:
                signals[i] = 0.0
    
    return signals