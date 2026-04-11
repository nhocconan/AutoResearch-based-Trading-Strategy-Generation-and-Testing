#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based stoploss
# - Long: price breaks above 20-period Donchian high with volume > 1.5x 20-period average
# - Short: price breaks below 20-period Donchian low with volume > 1.5x 20-period average
# - Exit: ATR trailing stop (3x ATR from highest high/lowest low since entry)
# - Works in bull markets via breakouts and in bear markets via short breakdowns
# - Volume confirmation filters false breakouts, ATR stop manages risk
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits

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
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Pre-compute 20-period Donchian channels
    high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 20-period ATR for volatility and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute 20-period volume average for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(20, n):  # Start after 20-bar warmup for indicators
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
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Donchian levels
        donchian_high = high_rolling_max[i]
        donchian_low = low_rolling_min[i]
        
        # ATR for stoploss
        atr_value = atr[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price breaks above Donchian high with volume confirmation
        if close_price > donchian_high and vol_confirm:
            enter_long = True
        
        # Short breakout: price breaks below Donchian low with volume confirmation
        if close_price < donchian_low and vol_confirm:
            enter_short = True
        
        # Exit conditions: ATR trailing stop
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Update highest high since entry
            if high_price > highest_high_since_entry:
                highest_high_since_entry = high_price
            # Exit long if price drops 3x ATR from highest high
            if close_price < highest_high_since_entry - 3.0 * atr_value:
                exit_long = True
        elif position == -1:
            # Update lowest low since entry
            if low_price < lowest_low_since_entry:
                lowest_low_since_entry = low_price
            # Exit short if price rises 3x ATR from lowest low
            if close_price > lowest_low_since_entry + 3.0 * atr_value:
                exit_short = True
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            entry_price = close_price
            highest_high_since_entry = high_price
            lowest_low_since_entry = low_price
            signals[i] = 0.30
        elif enter_short and position != -1:
            position = -1
            entry_price = close_price
            highest_high_since_entry = high_price
            lowest_low_since_entry = low_price
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