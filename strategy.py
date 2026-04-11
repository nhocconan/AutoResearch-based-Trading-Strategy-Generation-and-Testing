#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and ATR-based trailing stop
# - Long: Close breaks above Donchian upper band (20-period high) + volume > 1.5x 20-period average
# - Short: Close breaks below Donchian lower band (20-period low) + volume > 1.5x 20-period average
# - Exit: ATR trailing stop (highest high since entry - 3.0*ATR for longs, lowest low since entry + 3.0*ATR for shorts)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits
# - Donchian channels provide clear breakout levels in trending markets
# - Volume confirmation filters false breakouts
# - ATR trailing stop allows profits to run while controlling risk

name = "4h_1d_donchian_volume_atrstop_v1"
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
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d volume SMA(20)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute Donchian channels on 4h timeframe
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute ATR for trailing stop (14-period)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        high_price = high[i]
        low_price = low[i]
        
        # Donchian levels
        upper_band = highest_high_20[i]
        lower_band = lowest_low_20[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price closes above upper Donchian band with volume confirmation
        if close_price > upper_band and vol_confirm:
            enter_long = True
        
        # Short breakout: price closes below lower Donchian band with volume confirmation
        if close_price < lower_band and vol_confirm:
            enter_short = True
        
        # Exit conditions: ATR trailing stop
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Update highest high since entry
            if i == 50 or position != 1:  # Reset on new position
                highest_since_entry = high_price
            else:
                highest_since_entry = max(highest_since_entry, high_price)
            # Exit if price drops 3*ATR below highest high since entry
            exit_long = close_price <= highest_since_entry - 3.0 * atr_14[i]
        elif position == -1:
            # Update lowest low since entry
            if i == 50 or position != -1:  # Reset on new position
                lowest_since_entry = low_price
            else:
                lowest_since_entry = min(lowest_since_entry, low_price)
            # Exit if price rises 3*ATR above lowest low since entry
            exit_short = close_price >= lowest_since_entry + 3.0 * atr_14[i]
        
        # Track entry price for reference
        if enter_long or enter_short:
            entry_price = close_price
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            highest_since_entry = high_price
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            lowest_since_entry = low_price
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