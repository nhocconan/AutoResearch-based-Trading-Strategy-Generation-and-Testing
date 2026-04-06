#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter and volume confirmation
# Enter long when: price breaks above Donchian upper band (20-period high), price > 1d EMA(50), volume > 1.5x 20-period avg
# Enter short when: price breaks below Donchian lower band (20-period low), price < 1d EMA(50), volume > 1.5x 20-period avg
# Exit via ATR(14) trailing stop: long exits when price < highest_high_since_entry - 3*ATR, short exits when price > lowest_low_since_entry + 3*ATR
# Donchian breakouts capture momentum, EMA filter ensures trend alignment, volume confirms strength
# Target: 75-200 total trades over 4 years (19-50/year)

name = "4h_donchian20_1dema_vol_atrstop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_band = high_roll
    lower_band = low_roll
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    # ATR(14) for stop loss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = tr1.iloc[0]  # First bar: no previous close
    tr3.iloc[0] = tr1.iloc[0]
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = -1
    highest_since_entry = 0
    lowest_since_entry = 0
    
    for i in range(20, n):  # Wait for indicators to stabilize
        # Skip if required data not available
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Update highest high since entry
            if i == entry_bar:
                highest_since_entry = high[i]
            else:
                highest_since_entry = max(highest_since_entry, high[i])
            
            # Exit: ATR trailing stop
            if high[i] - low[i] > 0:  # Avoid division by zero in ATR
                stop_level = highest_since_entry - 3.0 * atr[i]
                if close[i] < stop_level:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
                
        elif position == -1:  # short position
            # Update lowest low since entry
            if i == entry_bar:
                lowest_since_entry = low[i]
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
            
            # Exit: ATR trailing stop
            if high[i] - low[i] > 0:  # Avoid division by zero in ATR
                stop_level = lowest_since_entry + 3.0 * atr[i]
                if close[i] > stop_level:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
                
        else:
            # Look for entries: Donchian breakout + trend filter + volume
            if volume[i] > volume_threshold[i]:
                # Long breakout: price closes above upper band
                if close[i] > upper_band[i] and close[i] > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_bar = i
                    highest_since_entry = high[i]
                # Short breakout: price closes below lower band
                elif close[i] < lower_band[i] and close[i] < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_bar = i
                    lowest_since_entry = low[i]
    
    return signals