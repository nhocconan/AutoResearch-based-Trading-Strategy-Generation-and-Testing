#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian channel breakout with 1d ATR filter and volume confirmation
# Donchian(20) identifies breakouts from recent price channels, effective in both trending and ranging markets
# 1d ATR filter ensures volatility is elevated (ATR > 1.2x 20-period average) to avoid false breakouts in low vol
# Volume confirmation (1.8x 20-period average) validates breakout conviction
# Exits on opposite Donchian(10) breakout to capture mean reversion in ranging markets
# Targets 50-150 trades over 4 years (12-37/year) for 6h timeframe
# Works in bull markets via trend continuation breaks and bear markets via mean reversion exits

name = "6h_Donchian20_1dATR_Filter_Volume_Confirmation_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(20) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with 1d index
    
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_ma_20 = pd.Series(atr_20).rolling(window=20, min_periods=1).mean().values  # 20-period average of ATR
    atr_filter = atr_20 > (atr_ma_20 * 1.2)  # Elevated volatility filter
    
    atr_filter_aligned = align_htf_to_ltf(prices, df_1d, atr_filter)
    
    # Calculate Donchian channels from 6h data
    donchian_len = 20
    donchian_exit_len = 10
    
    # Upper channel: highest high over period
    upper_channel = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().values
    
    # Exit channels (shorter period for mean reversion)
    upper_exit = pd.Series(high).rolling(window=donchian_exit_len, min_periods=donchian_exit_len).max().values
    lower_exit = pd.Series(low).rolling(window=donchian_exit_len, min_periods=donchian_exit_len).min().values
    
    # Volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian and volume MA)
    start_idx = max(donchian_len, 20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(upper_exit[i]) or np.isnan(lower_exit[i]) or 
            np.isnan(atr_filter_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian upper + volatility filter + volume spike
            if close[i] > upper_channel[i] and atr_filter_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower + volatility filter + volume spike
            elif close[i] < lower_channel[i] and atr_filter_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Donchian lower exit (mean reversion) or opposite Donchian break
            if close[i] < lower_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian upper exit (mean reversion) or opposite Donchian break
            if close[i] > upper_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals