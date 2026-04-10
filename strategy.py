#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 12h trend filter and volume confirmation
# - Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# - Long: Bull Power > 0 AND Bear Power rising (less negative) in 12h uptrend
# - Short: Bear Power < 0 AND Bull Power falling (less positive) in 12h downtrend
# - Volume filter: 6h volume > 1.5x 20-period average to confirm momentum
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 2.5x ATR(20) on 6h
# - Works in bull/bear: Measures bull/bear power directly; trend filter avoids counter-trend

name = "6h_12h_elder_ray_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h EMA(20) for trend filter
    close_12h = df_12h['close'].values
    ema_20 = pd.Series(close_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_12h, ema_20)
    
    # Pre-compute 6h EMA(13) for Elder Ray
    close_6h = prices['close'].values
    ema_13 = pd.Series(close_6h).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Pre-compute 6h Elder Ray components
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    bull_power = high_6h - ema_13
    bear_power = low_6h - ema_13
    
    # Pre-compute 6h volume filter
    volume_6h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (1.5 * avg_volume_20)
    
    # Pre-compute 6h ATR(20) for stoploss
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_spike[i]) or 
            np.isnan(atr_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Bull Power <= 0 OR Bear Power >= 0 (momentum lost) OR stoploss hit
            if bull_power[i] <= 0 or bear_power[i] >= 0 or close_6h[i] < entry_price - 2.5 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bear Power >= 0 OR Bull Power <= 0 (momentum lost) OR stoploss hit
            if bear_power[i] >= 0 or bull_power[i] <= 0 or close_6h[i] > entry_price + 2.5 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Elder Ray signals with trend and volume filters
            if vol_spike[i]:
                # Long: Bull Power > 0 AND Bear Power rising (less negative) in 12h uptrend
                if (bull_power[i] > 0 and bear_power[i] < 0 and 
                    close_6h[i] > ema_20_aligned[i] and bear_power[i] > bear_power[i-1]):
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                # Short: Bear Power < 0 AND Bull Power falling (less positive) in 12h downtrend
                elif (bear_power[i] < 0 and bull_power[i] > 0 and 
                      close_6h[i] < ema_20_aligned[i] and bull_power[i] < bull_power[i-1]):
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
    
    return signals