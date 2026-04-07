#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian(20) Breakout + 1d Volume Spike + ATR Volatility Filter
# Hypothesis: Donchian breakouts capture strong trends, volume spikes confirm institutional participation,
# and ATR filter ensures trades occur in sufficient volatility. Works in bull via upside breakouts,
# in bear via downside breakouts. Target: 20-50 trades/year (80-200 total over 4 years) for 4h timeframe.

name = "4h_donchian_20_volume_atr_v1"
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
    
    # Get 1d data for volume and volatility filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d Volume Spike: volume > 2x 20-day average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=10).mean().values
    vol_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 1d ATR(14) for volatility filter
    tr1 = high_1d[1:] - low_1d[:-1]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 4h Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR > 0.5 * 50-period average ATR (avoid low volatility chop)
        if i >= 50:
            atr_ma_50 = np.nanmean(atr_1d_aligned[i-50:i]) if not np.isnan(np.nanmean(atr_1d_aligned[i-50:i])) else 0
            vol_filter = atr_1d_aligned[i] > (0.5 * atr_ma_50) if atr_ma_50 > 0 else False
        else:
            vol_filter = False
        
        # Volume confirmation
        vol_ok = vol_spike_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band or volatility drops
            if close[i] < low_20[i] or not vol_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band or volatility drops
            if close[i] > high_20[i] or not vol_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            if vol_ok and vol_filter:
                # Long breakout: price closes above Donchian upper band
                if close[i] > high_20[i]:
                    position = 1
                    signals[i] = 0.30
                # Short breakout: price closes below Donchian lower band
                elif close[i] < low_20[i]:
                    position = -1
                    signals[i] = -0.30
    
    return signals