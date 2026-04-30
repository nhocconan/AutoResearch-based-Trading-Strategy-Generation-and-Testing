#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ATR regime filter and volume confirmation
# Donchian breakouts capture momentum shifts in both bull and bear markets
# 1d ATR regime filter: only trade when ATR(14) > 20-period ATR median (high volatility regimes)
# Volume confirmation (>1.5x average) ensures breakout legitimacy
# This combination reduces false breakouts during low volatility sideways markets
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "6h_Donchian20_ATRRegime_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h Donchian channels (20-period)
    # Upper band = highest high of last 20 periods
    # Lower band = lowest low of last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Need previous bar's bands to avoid look-ahead
    donchian_upper_prev = np.roll(donchian_upper, 1)
    donchian_lower_prev = np.roll(donchian_lower, 1)
    donchian_upper_prev[0] = np.nan
    donchian_lower_prev[0] = np.nan
    
    # Breakout conditions
    breakout_up = close > donchian_upper_prev
    breakout_down = close < donchian_lower_prev
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # Calculate 1d ATR(14) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # True Range calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar
    tr2[0] = np.abs(high_1d[0] - close_1d[-1]) if len(close_1d) > 1 else high_1d[0] - low_1d[0]
    tr3[0] = np.abs(low_1d[0] - close_1d[-1]) if len(close_1d) > 1 else high_1d[0] - low_1d[0]
    
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR regime: only trade when current ATR > 20-period median ATR (high vol regime)
    atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    atr_median_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).median().values
    atr_regime = atr_14 > atr_median_20
    
    # Align 1d indicators to 6h timeframe
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime)
    atr_median_20_aligned = align_htf_to_ltf(prices, df_1d, atr_median_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_upper_prev[i]) or 
            np.isnan(donchian_lower_prev[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(atr_regime_aligned[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_breakout_up = breakout_up[i]
        curr_breakout_down = breakout_down[i]
        curr_volume_confirm = volume_confirm[i]
        curr_atr_regime = atr_regime_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on breakout with volume confirmation and high volatility regime
            if curr_volume_confirm and curr_atr_regime:
                # Bullish breakout: price above Donchian upper band
                if curr_breakout_up:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price below Donchian lower band
                elif curr_breakout_down:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price closes below Donchian lower band (reversal) or above upper band (take profit)
            if curr_close < donchian_lower_prev[i] or curr_close > donchian_upper_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band (reversal) or below lower band (take profit)
            if curr_close > donchian_upper_prev[i] or curr_close < donchian_lower_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals