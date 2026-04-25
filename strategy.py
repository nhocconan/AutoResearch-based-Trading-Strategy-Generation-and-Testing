#!/usr/bin/env python3
"""
6h_WeeklyDonchian_With_1dVolumeSpike_and_ChopFilter
Hypothesis: 6-hour Donchian(20) breakout confirmed by 1-day volume spike (>2.0x 20-period average) and filtered by 6-hour chopiness index (CHOP < 38.2 = trending, CHOP > 61.8 = range). 
Long when price breaks above 20-period 6h high in trending regime with volume confirmation. 
Short when price breaks below 20-period 6h low in trending regime with volume confirmation.
Exit via opposite Donchian boundary or ATR trailing stop (2.5*ATR from extreme).
Volume conviction + trend filter (CHOP) reduces false breakouts in choppy/range markets. 
Designed for ~80-150 trades over 4 years (20-37/year) via tight Donchian breakout + volume + trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume spike filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (2.0 * vol_ma_20_1d)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # ATR for stoploss (14-period) on 6h
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Donchian channels (20-period) on 6h
    donch_len = 20
    donch_high = pd.Series(high).rolling(window=donch_len, min_periods=donch_len).max().values
    donch_low = pd.Series(low).rolling(window=donch_len, min_periods=donch_len).min().values
    
    # Chopiness Index (14-period) on 6h to filter ranging markets
    chop_len = 14
    atr_chop = pd.Series(tr).rolling(window=chop_len, min_periods=chop_len).sum().values
    hh = pd.Series(high).rolling(window=chop_len, min_periods=chop_len).max().values
    ll = pd.Series(low).rolling(window=chop_len, min_periods=chop_len).min().values
    chop_denom = hh - ll
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid div0
    chop = 100 * np.log10(atr_chop / chop_denom * np.sqrt(chop_len)) / np.log10(chop_len)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0
    short_extreme = 0.0
    
    # Start index: need warmup for all indicators
    start_idx = max(100, atr_period, donch_len, chop_len, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(atr[i]) or np.isnan(chop[i]) or np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Trending regime: CHOP < 38.2 (strong trend) OR CHOP > 61.8 (strong range - we avoid)
            # Actually we want trending only: CHOP < 38.2
            is_trending = chop[i] < 38.2
            if not is_trending:
                signals[i] = 0.0
                continue
            
            # Volume confirmation from 1d
            vol_confirmed = vol_spike_aligned[i]
            
            if vol_confirmed:
                # Long: break above Donchian high
                if close[i] > donch_high[i]:
                    signals[i] = 0.25
                    position = 1
                    long_extreme = close[i]
                # Short: break below Donchian low
                elif close[i] < donch_low[i]:
                    signals[i] = -0.25
                    position = -1
                    short_extreme = close[i]
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Update extreme for trailing stop
            if close[i] > long_extreme:
                long_extreme = close[i]
            # Exit conditions:
            # 1. ATR trailing stop (2.5*ATR from extreme)
            atr_stop = long_extreme - 2.5 * atr[i]
            # 2. Price breaks below Donchian low (opposite boundary)
            if close[i] <= atr_stop or close[i] < donch_low[i]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Update extreme for trailing stop
            if close[i] < short_extreme:
                short_extreme = close[i]
            # Exit conditions:
            # 1. ATR trailing stop (2.5*ATR from extreme)
            atr_stop = short_extreme + 2.5 * atr[i]
            # 2. Price breaks above Donchian high (opposite boundary)
            if close[i] >= atr_stop or close[i] > donch_high[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyDonchian_With_1dVolumeSpike_and_ChopFilter"
timeframe = "6h"
leverage = 1.0