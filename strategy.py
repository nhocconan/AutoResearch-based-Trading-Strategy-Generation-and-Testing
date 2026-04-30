#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ATR volatility filter and volume confirmation (>1.5x average)
# Uses 12h timeframe to minimize trade frequency (target: 50-150 total trades over 4 years)
# 1d ATR(14) measures volatility regime: only trade when ATR > its 50-period MA (expanding volatility)
# Volume confirmation >1.5x 20-period average avoids low-conviction breakouts
# Discrete position sizing: 0.25 for entries to balance return and drawdown
# Works in all markets: Donchian breakouts capture trends, volatility filter avoids choppy false signals

name = "12h_Donchian20_ATR_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian channels (20-period) from previous bar
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Need previous bar's levels to avoid look-ahead
    highest_20_prev = np.roll(highest_20, 1)
    lowest_20_prev = np.roll(lowest_20, 1)
    highest_20_prev[0] = np.nan
    lowest_20_prev[0] = np.nan
    
    # Breakout conditions
    breakout_up = close > highest_20_prev
    breakout_down = close < lowest_20_prev
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # Calculate 1d ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    
    # Volatility filter: only trade when current ATR > its 50-period MA (expanding volatility)
    volatility_filter = atr_14_aligned > atr_ma_50_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20, 14, 50)  # warmup for Donchian (20), ATR (14), ATR MA (50)
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(highest_20_prev[i]) or 
            np.isnan(lowest_20_prev[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(atr_14_aligned[i]) or
            np.isnan(atr_ma_50_aligned[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_breakout_up = breakout_up[i]
        curr_breakout_down = breakout_down[i]
        curr_volume_confirm = volume_confirm[i]
        curr_volatility_filter = volatility_filter[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on breakout with volume confirmation and volatility filter
            if curr_volume_confirm and curr_volatility_filter:
                # Bullish breakout: price above Donchian upper band
                if curr_breakout_up:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price below Donchian lower band
                elif curr_breakout_down:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price closes below Donchian lower band (reversal)
            if curr_close < lowest_20_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band (reversal)
            if curr_close > highest_20_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals