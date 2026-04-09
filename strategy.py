#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume spike + ATR trailing stop
# - Primary signal: Price breaks above/below 20-period 12h Donchian channel
# - Volume confirmation: 1d volume > 1.5x 20-period average volume (avoid low-participation breakouts)
# - ATR trailing stop: Exit when price retraces 2.0x ATR from extreme
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: Donchian breakouts capture trends, volume filter ensures validity, ATR stop manages risk in volatile markets

name = "12h_1d_donchian_volume_atr_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d volume average for confirmation
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_avg_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_20)
    
    # Pre-compute 12h Donchian channels (20-period)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    highest_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute ATR(14) for trailing stop
    tr1 = pd.Series(high_12h).diff().abs()
    tr2 = (pd.Series(high_12h) - pd.Series(close_12h).shift()).abs()
    tr3 = (pd.Series(low_12h) - pd.Series(close_12h).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    long_extreme = 0.0   # highest high since long entry
    short_extreme = 0.0  # lowest low since short entry
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(volume_avg_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        current_volume = df_1d['volume'].iloc[i] if i < len(df_1d) else volume_1d[-1]
        
        if position == 1:  # Long position
            # Update long extreme
            long_extreme = max(long_extreme, high_12h[i])
            # ATR trailing stop: exit if price retraces 2.0*ATR from extreme
            if low_12h[i] < long_extreme - 2.0 * atr[i]:
                position = 0
                long_extreme = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update short extreme
            short_extreme = min(short_extreme, low_12h[i])
            # ATR trailing stop: exit if price retraces 2.0*ATR from extreme
            if high_12h[i] > short_extreme + 2.0 * atr[i]:
                position = 0
                short_extreme = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Volume confirmation: 1d volume > 1.5x 20-period average
            volume_ok = current_volume > 1.5 * volume_avg_aligned[i]
            
            # Look for Donchian breakout with volume confirmation
            # Long: price breaks above highest high (20-period) AND volume confirmation
            if close_12h[i] > highest_high[i] and volume_ok:
                position = 1
                long_extreme = high_12h[i]
                signals[i] = 0.25
            # Short: price breaks below lowest low (20-period) AND volume confirmation
            elif close_12h[i] < lowest_low[i] and volume_ok:
                position = -1
                short_extreme = low_12h[i]
                signals[i] = -0.25
    
    return signals