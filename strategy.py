#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d ATR volume filter and chop regime filter
# - Uses 12h Donchian(20) breakouts for entries in direction of 1d trend (EMA50)
# - Volume confirmation: 12h volume > 1.3x 20-period average AND 1d volume > 1.2x 20-period average
# - Regime filter: 12h Choppiness Index < 38.2 (trending) to avoid false breakouts in sideways markets
# - ATR(14) trailing stop at 2.0x ATR from extreme for risk control
# - Position size: 0.25 (25% of capital) - discrete level to minimize fee churn
# - Target: ~12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in both bull/bear: Donchian captures breakouts, chop filter avoids whipsaws, multi-timeframe volume confirms strength

name = "12h_1d_donchian_volume_chop_v3"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate 1d volume > 1.2x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (1.2 * avg_volume_20_1d)
    
    # Align 1d indicators to 12h timeframe (completed 1d bar only)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d)
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h volume > 1.3x 20-period average
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * avg_volume_20)
    
    # 12h ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 12h Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high,n) - min(low,n))) / log10(n)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    # Avoid division by zero
    chop = np.where(range_14 > 0, 100 * np.log10(atr_14 / range_14) / np.log10(14), 50)
    chop = np.where(np.isnan(chop), 50, chop)
    chop_regime = chop < 38.2  # True when trending (avoid false breakouts in ranging markets)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(volume_confirm_1d_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(chop_regime[i]) or
            atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.0x ATR from high
            if low[i] <= highest_since_entry - (2.0 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.0x ATR from low
            if high[i] >= lowest_since_entry + (2.0 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and trend filter
            # Long: price breaks above Donchian upper band AND volume spike AND 1d volume confirm AND 1d price > EMA50 AND chop regime trending
            if (high[i] >= highest_20[i] and 
                volume_spike[i] and 
                volume_confirm_1d_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and
                chop_regime[i]):
                position = 1
                highest_since_entry = high[i]
                lowest_since_entry = high[i]
                signals[i] = 0.25
            # Short: price breaks below Donchian lower band AND volume spike AND 1d volume confirm AND 1d price < EMA50 AND chop regime trending
            elif (low[i] <= lowest_20[i] and 
                  volume_spike[i] and 
                  volume_confirm_1d_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and
                  chop_regime[i]):
                position = -1
                highest_since_entry = low[i]
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals