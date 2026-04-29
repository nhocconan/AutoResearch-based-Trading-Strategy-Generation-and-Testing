#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with volume confirmation and ATR trailing stop
# Long when price breaks above 20-period Donchian high AND volume > 1.5x 20-period average
# Short when price breaks below 20-period Donchian low AND volume > 1.5x 20-period average
# Uses ATR-based trailing stop (2.5x ATR) for risk management
# Discrete position sizing (0.25) to minimize fee drag while maintaining profitability
# Target: 15-25 trades/year on 4h timeframe (~60-100 total over 4 years) to avoid overtrading
# Works in bull markets via long breakouts with volume confirmation
# Works in bear markets via short breakdowns with volume confirmation
# Uses proven structure from top performers: price channel breakout + volume + ATR stop

name = "4h_Donchian_Breakout_Volume_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    # Upper channel: highest high of last 20 periods
    # Lower channel: lowest low of last 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = 20  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_highest_high = highest_high[i]
        curr_lowest_low = lowest_low[i]
        curr_vol_spike = vol_spike[i]
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            # Trailing stop: 2.5 * ATR below highest high
            stop_price = highest_high_since_entry - 2.5 * curr_atr
            # Exit conditions: price below trailing stop
            if curr_close < stop_price:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            # Trailing stop: 2.5 * ATR above lowest low
            stop_price = lowest_low_since_entry + 2.5 * curr_atr
            # Exit conditions: price above trailing stop
            if curr_close > stop_price:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper channel AND volume spike
            if curr_close > curr_highest_high and curr_vol_spike:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = curr_high
            # Short entry: price breaks below Donchian lower channel AND volume spike
            elif curr_close < curr_lowest_low and curr_vol_spike:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals