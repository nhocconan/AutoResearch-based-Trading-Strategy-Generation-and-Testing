#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_alligator_vortex_volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly trend filter ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Williams Alligator (12h) - using SMAs
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Vortex Indicator (12h)
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = np.nan
    vm_minus[0] = np.nan
    vi_plus = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum() / pd.Series(tr).rolling(window=14, min_periods=14).sum()
    vi_minus = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum() / pd.Series(tr).rolling(window=14, min_periods=14).sum()
    vi_plus = vi_plus.values
    vi_minus = vi_minus.values
    
    # Volume confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Need indicators warmed up
    
    for i in range(start_idx, n):
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or np.isnan(avg_volume[i]) or np.isnan(ema50_1w_aligned[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema50_1w_aligned[i]
        weekly_downtrend = close[i] < ema50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Alligator lines cross (teeth < lips) OR Vortex reversal
            if teeth[i] < lips[i] or (vi_plus[i] < vi_minus[i] and vi_plus[i-1] >= vi_minus[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator lines cross (teeth > lips) OR Vortex reversal
            if teeth[i] > lips[i] or (vi_minus[i] < vi_plus[i] and vi_minus[i-1] >= vi_plus[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            if volume_ok:
                # Long: Alligator aligned (lips > teeth > jaw) AND VI+ > VI- AND weekly uptrend
                if lips[i] > teeth[i] and teeth[i] > jaw[i] and vi_plus[i] > vi_minus[i] and weekly_uptrend:
                    position = 1
                    signals[i] = 0.25
                # Short: Alligator aligned (jaw > teeth > lips) AND VI- > VI+ AND weekly downtrend
                elif jaw[i] > teeth[i] and teeth[i] > lips[i] and vi_minus[i] > vi_plus[i] and weekly_downtrend:
                    position = -1
                    signals[i] = -0.25
    
    return signals