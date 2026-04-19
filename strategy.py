#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h trend following using daily Supertrend with volume confirmation.
# Supertrend adapts to volatility via ATR, performing well in trending markets.
# Volume filter ensures breakouts have conviction, reducing false signals.
# Designed for 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
# Works in bull (captures trends) and bear (avoids false breaks via volume filter).

name = "6h_Supertrend_Volume_Confirmation_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Supertrend calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR (10-period) for Supertrend
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    atr_10_1d = pd.Series(tr1).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend parameters
    atr_mult = 3.0
    
    # Basic upper and lower bands
    basic_ub = (high_1d + low_1d) / 2 + atr_mult * atr_10_1d
    basic_lb = (high_1d + low_1d) / 2 - atr_mult * atr_10_1d
    
    # Initialize final bands
    final_ub = np.full_like(close_1d, np.nan)
    final_lb = np.full_like(close_1d, np.nan)
    
    # Calculate final bands
    for i in range(1, len(close_1d)):
        if np.isnan(basic_ub[i]) or np.isnan(basic_lb[i]):
            final_ub[i] = final_ub[i-1]
            final_lb[i] = final_lb[i-1]
        else:
            if close_1d[i-1] <= final_ub[i-1]:
                final_ub[i] = min(basic_ub[i], final_ub[i-1])
            else:
                final_ub[i] = basic_ub[i]
            
            if close_1d[i-1] >= final_lb[i-1]:
                final_lb[i] = max(basic_lb[i], final_lb[i-1])
            else:
                final_lb[i] = basic_lb[i]
    
    # Determine Supertrend direction
    supertrend = np.full_like(close_1d, np.nan)
    for i in range(1, len(close_1d)):
        if np.isnan(final_ub[i]) or np.isnan(final_lb[i]):
            supertrend[i] = supertrend[i-1]
        else:
            if supertrend[i-1] == final_ub[i-1]:
                if close_1d[i] <= final_ub[i]:
                    supertrend[i] = final_ub[i]
                else:
                    supertrend[i] = final_lb[i]
            else:
                if close_1d[i] >= final_lb[i]:
                    supertrend[i] = final_lb[i]
                else:
                    supertrend[i] = final_ub[i]
    
    # Align Supertrend to 6h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    
    # Volume confirmation: current volume > 1.5x 20-period average (6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(supertrend_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        st = supertrend_aligned[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: price above Supertrend with volume confirmation
            if price > st and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price below Supertrend with volume confirmation
            elif price < st and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price below Supertrend
            if price < st:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price above Supertrend
            if price > st:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals