#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian20 + Volume + ADX Trend Filter
# Hypothesis: Donchian breakouts with volume confirmation and ADX trend filter capture strong moves
# while avoiding choppy markets. Works in bull via upper band breaks, in bear via lower band breaks.
# Uses ADX > 25 to filter for trending markets only, reducing false signals and trade frequency.
# Target: 20-50 trades/year to stay within optimal range for 4h timeframe.
name = "4h_donchian20_volume_adx_v16"
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
    
    # Volume confirmation: current volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ADX calculation (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_sum = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_sum = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_sum / tr_sum
    minus_di = 100 * minus_dm_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        
        # ADX trend filter: only trade when ADX > 25 (trending market)
        trend_filter = adx[i] > 25
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band or trend weakens
            if close[i] < low_roll[i] or not trend_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band or trend weakens
            if close[i] > high_roll[i] or not trend_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price closes above Donchian upper band with volume and trend
            if close[i] > high_roll[i] and vol_confirm and trend_filter:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below Donchian lower band with volume and trend
            elif close[i] < low_roll[i] and vol_confirm and trend_filter:
                position = -1
                signals[i] = -0.25
    
    return signals