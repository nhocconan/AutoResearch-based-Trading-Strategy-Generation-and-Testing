# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout (20) + 1d volume spike (2x 20-period avg) + ADX filter (20-40 for trending).
# Donchian provides price channel structure, volume confirms breakout strength, ADX ensures trending market (not chop).
# Designed for 12h timeframe to capture medium-term trends with low trade frequency (target 15-30/year).
# Entry: Price breaks above/below 20-period Donchian channel + volume spike + ADX between 20 and 40.
# Exit: Price returns to Donchian midpoint (mean reversion) OR ADX drops below 20 (trend weakening).
# Uses strict conditions to limit trades and avoid overtrading.

name = "12h_Donchian_Volume_ADX"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period high/low)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume filter: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    # ADX calculation (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR and DM (14-period Wilder's smoothing)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilders_smooth(tr, 14)
    plus_di = 100 * wilders_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilders_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smooth(dx, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(volume_ma[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: breakout above Donchian high + volume spike + ADX trending (20-40)
            if (close[i] > donchian_high[i] and 
                volume_spike[i] and 
                adx[i] >= 20 and adx[i] <= 40):
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low + volume spike + ADX trending (20-40)
            elif (close[i] < donchian_low[i] and 
                  volume_spike[i] and 
                  adx[i] >= 20 and adx[i] <= 40):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price returns to midpoint OR ADX drops below 20 (trend weakening)
            if (close[i] <= donchian_mid[i]) or (adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price returns to midpoint OR ADX drops below 20 (trend weakening)
            if (close[i] >= donchian_mid[i]) or (adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals