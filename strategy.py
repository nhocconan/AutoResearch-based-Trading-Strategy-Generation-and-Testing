#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + chop regime filter
# Donchian breakout captures strong momentum moves, volume spike confirms participation,
# chop regime (CHOP > 61.8) filters for ranging markets where we mean revert at Donchian bands
# Works in both bull and bear markets by adapting to regime: trend follow in trending (CHOP < 38.2),
# mean revert in ranging (CHOP > 61.8). Uses 1d for HTF regime and volume confirmation.

name = "4h_Donchian20_1dVolumeSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for regime filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d True Range for Chopiness Index
    tr1 = np.abs(df_1d['high'].values - df_1d['low'].values)
    tr2 = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
    tr3 = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate Chopiness Index: CHOP = 100 * log10(sum(TR,14) / (max(HH,14) - min(LL,14))) / log10(14)
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    denominator = hh - ll
    chop_raw = 100 * np.log10(tr_sum / (denominator + 1e-10)) / np.log10(14)
    chop = pd.Series(chop_raw).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d volume MA for spike detection
    vol_ma = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (vol_ma * 2.0)  # 2x volume spike
    
    # Align 1d indicators to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_period = 20
    dc_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    dc_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    dc_mid = (dc_high + dc_low) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian, chop and volume MA)
    start_idx = max(donchian_period, 30) + 5
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        chop_val = chop_aligned[i]
        vol_spike = volume_spike_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Determine regime from Chopiness Index
            trending = chop_val < 38.2   # Strong trend
            ranging = chop_val > 61.8    # Strong ranging
            transition = not (trending or ranging)  # Weak trend/range
            
            if trending and vol_spike:
                # In trending market with volume spike: follow Donchian breakout
                if close[i] > dc_high[i-1]:  # Bullish breakout
                    signals[i] = 0.30
                    position = 1
                elif close[i] < dc_low[i-1]:  # Bearish breakout
                    signals[i] = -0.30
                    position = -1
                else:
                    signals[i] = 0.0
            elif ranging and vol_spike:
                # In ranging market with volume spike: mean revert at Donchian bands
                if close[i] < dc_low[i-1]:  # Oversold bounce
                    signals[i] = 0.30
                    position = 1
                elif close[i] > dc_high[i-1]:  # Overbought fade
                    signals[i] = -0.30
                    position = -1
                else:
                    signals[i] = 0.0
            else:  # transition regime or no volume spike
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions
            exit_signal = False
            if trending:
                # Exit trending long when price crosses below Donchian mid
                if close[i] < dc_mid[i]:
                    exit_signal = True
            else:
                # Exit ranging long when price reaches Donchian high (mean reversion target)
                if close[i] >= dc_high[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit conditions
            exit_signal = False
            if trending:
                # Exit trending short when price crosses above Donchian mid
                if close[i] > dc_mid[i]:
                    exit_signal = True
            else:
                # Exit ranging short when price reaches Donchian low (mean reversion target)
                if close[i] <= dc_low[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals