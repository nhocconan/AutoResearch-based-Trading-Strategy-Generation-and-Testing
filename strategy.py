#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + ATR filter for stoploss
# - Long: price breaks above 20-period high with volume > 1.5x 20-period average volume
# - Short: price breaks below 20-period low with volume > 1.5x 20-period average volume
# - Exit: ATR-based trailing stop (3 * ATR) from highest high/lowest low since entry
# - Works in bull markets via breakouts and bear markets via short breakdowns
# - Target: 19-50 trades/year (75-200 total over 4 years) to stay within fee drag limits

name = "4h_donchian_breakout_volume_atr_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_high = 0.0
    lowest_low = 0.0
    
    # Pre-compute Donchian channels (20-period)
    high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute ATR (14-period) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(20, n):  # Start after 20-bar warmup for Donchian
        # Skip if any required data is invalid
        if (np.isnan(high_rolling_max[i]) or np.isnan(low_rolling_min[i]) or
            np.isnan(atr[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Donchian levels
        upper_channel = high_rolling_max[i]
        lower_channel = low_rolling_min[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # ATR for stoploss
        atr_value = atr[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price breaks above upper Donchian with volume confirmation
        if close_price > upper_channel and vol_confirm:
            enter_long = True
        
        # Short breakout: price breaks below lower Donchian with volume confirmation
        if close_price < lower_channel and vol_confirm:
            enter_short = True
        
        # Exit conditions: ATR-based trailing stop
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Update highest high since entry
            if close_price > highest_high:
                highest_high = close_price
            # Exit long if price drops to highest_high - 3 * ATR
            if close_price <= highest_high - 3.0 * atr_value:
                exit_long = True
        elif position == -1:
            # Update lowest low since entry
            if close_price < lowest_low:
                lowest_low = close_price
            # Exit short if price rises to lowest_low + 3 * ATR
            if close_price >= lowest_low + 3.0 * atr_value:
                exit_short = True
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            entry_price = close_price
            highest_high = close_price
            lowest_low = close_price
            signals[i] = 0.30
        elif enter_short and position != -1:
            position = -1
            entry_price = close_price
            highest_high = close_price
            lowest_low = close_price
            signals[i] = -0.30
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.30 if position == 1 else (-0.30 if position == -1 else 0.0)
    
    return signals