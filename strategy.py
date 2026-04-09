#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w Donchian breakout with volume confirmation and ATR filter
# Weekly Donchian channels (20-period) capture major trend structure that works in both bull and bear markets
# Breakouts above upper channel or below lower channel with volume confirmation signal trend continuation
# ATR filter ensures sufficient volatility to avoid choppy low-volume false breakouts
# Position size fixed at 0.25 (25%) to balance risk and return, using discrete levels to minimize fee churn
# Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1w_donchian_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Donchian channels (20-period)
    # Upper channel = highest high over past 20 periods
    # Lower channel = lowest low over past 20 periods
    high_ma_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w ATR (14-period) for volatility filtering
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align Donchian channels and ATR to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1w, high_ma_20)
    lower_aligned = align_htf_to_ltf(prices, df_1w, low_ma_20)
    atr_aligned = align_htf_to_ltf(prices, df_1w, atr_14)
    
    # Pre-compute volume confirmation (20-period average for 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or
            np.isnan(atr_aligned[i]) or np.isnan(vol_ma_20[i]) or
            atr_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average 12h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Volatility filter: only trade when ATR is above its 30-period average (avoid low-vol chop)
        atr_ma_30 = pd.Series(atr_aligned).rolling(window=30, min_periods=30).mean()
        if len(atr_ma_30) > i:
            vol_filter = atr_aligned[i] > atr_ma_30.iloc[i]
        else:
            vol_filter = True  # Not enough data for MA, allow trading
            
        if not vol_filter:
            signals[i] = 0.0
            continue
        
        # Fixed position size: 0.25 (25%) - discrete level to minimize fee churn
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit on retracement to lower Donchian channel
            if close[i] < lower_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on retracement to upper Donchian channel
            if close[i] > upper_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Donchian breakout with volume confirmation
            if volume_confirmed:
                # Breakout above upper channel -> long
                if close[i] > upper_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Breakout below lower channel -> short
                elif close[i] < lower_aligned[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals