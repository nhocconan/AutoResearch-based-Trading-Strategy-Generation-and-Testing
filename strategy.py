#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian20 + Volume + ADX Filter
# Hypothesis: Donchian breakouts with volume confirmation and ADX trend filter capture
# strong trending moves while avoiding whipsaws. Works in bull (breakouts up), bear (breakdowns down),
# and ranges (filtered by low ADX). Target: 20-50 trades/year to minimize fee drag.
name = "12h_donchian20_volume_adx_v15"
timeframe = "12h"
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
    
    # Donchian channels (20-period) on 12h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 12h volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ADX filter (14-period) to avoid choppy markets
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values (using Wilder's smoothing, equivalent to EMA with alpha=1/14)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr14 = wilders_smooth(tr, 14)
    plus_dm14 = wilders_smooth(plus_dm, 14)
    minus_dm14 = wilders_smooth(minus_dm, 14)
    
    # Avoid division by zero
    plus_di14 = np.where(tr14 != 0, (plus_dm14 / tr14) * 100, 0)
    minus_di14 = np.where(tr14 != 0, (minus_dm14 / tr14) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14) * 100, 0)
    adx = wilders_smooth(dx, 14)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        
        # ADX filter: only trade when trending (ADX > 20)
        trending = adx[i] > 20
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or trend weakens
            if close[i] < donchian_low[i] or not trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or trend weakens
            if close[i] > donchian_high[i] or not trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price closes above Donchian high with volume and trending up
            if close[i] > donchian_high[i] and vol_confirm and trending:
                # Additional filter: +DI > -DI for bullish bias
                if plus_di14[i] > minus_di14[i]:
                    position = 1
                    signals[i] = 0.30
            # Enter short: price closes below Donchian low with volume and trending down
            elif close[i] < donchian_low[i] and vol_confirm and trending:
                # Additional filter: -DI > +DI for bearish bias
                if minus_di14[i] > plus_di14[i]:
                    position = -1
                    signals[i] = -0.30
    
    return signals