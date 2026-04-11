#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + ATR trailing stop
# - Long: Price breaks above Donchian upper channel (20-bar high) with volume > 1.5x 20-bar average volume
# - Short: Price breaks below Donchian lower channel (20-bar low) with volume > 1.5x 20-bar average volume
# - Exit: ATR trailing stop (2.5 * ATR from highest high/lowest low since entry)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits
# - Donchian channels capture breakouts from consolidation with clear structure
# - Volume confirmation ensures institutional participation
# - ATR trailing stop adapts to volatility and locks in profits

name = "4h_donchian_volume_atrstop_v1"
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
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Pre-compute Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute ATR for trailing stop (14-period)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_sma_20[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Donchian breakout conditions
        breakout_up = close_price > highest_high[i]
        breakout_down = close_price < lowest_low[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price breaks above upper channel with volume confirmation
        if breakout_up and vol_confirm and position != 1:
            enter_long = True
        
        # Short breakout: price breaks below lower channel with volume confirmation
        if breakout_down and vol_confirm and position != -1:
            enter_short = True
        
        # Exit conditions: ATR trailing stop
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Update highest high since entry
            if i == 50 or position == 0:  # reset on new position
                highest_since_entry = close_price
            else:
                highest_since_entry = max(highest_since_entry, close_price)
            
            # Exit long if price drops 2.5 * ATR from highest high since entry
            exit_long = close_price <= highest_since_entry - 2.5 * atr_14[i]
        elif position == -1:
            # Update lowest low since entry
            if i == 50 or position == 0:  # reset on new position
                lowest_since_entry = close_price
            else:
                lowest_since_entry = min(lowest_since_entry, close_price)
            
            # Exit short if price rises 2.5 * ATR from lowest low since entry
            exit_short = close_price >= lowest_since_entry + 2.5 * atr_14[i]
        
        # Track entry price for reference
        if enter_long or enter_short:
            entry_price = close_price
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            highest_since_entry = close_price
            lowest_since_entry = close_price
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            highest_since_entry = close_price
            lowest_since_entry = close_price
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals