#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume spike + choppiness regime filter
# - Primary signal: Donchian channel breakout (20-period high/low) on 4h timeframe
# - Volume confirmation: 4h volume > 1.5 * 20-period median volume (avoid low-participation breakouts)
# - Regime filter: Choppiness Index(14) > 61.8 for ranging markets (mean reversion at channel edges),
#                  Choppiness Index(14) < 38.2 for trending markets (breakout continuation)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 20-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - Works in bull/bear: Choppiness regime adapts to market conditions - mean revert in range, follow trend in trending markets

name = "4h_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA50 for higher timeframe trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Primary timeframe (4h) data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period) on 4h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume regime: volume > 1.5 * 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.5 * median_volume_20)
    
    # Choppiness Index (14) - measures whether market is choppy (ranging) or trending
    # CHOP = 100 * log10(sum(ATR(14) over n periods) / log10(highest_high - lowest_low over n periods))
    tr1 = pd.Series(high).rolling(window=2, min_periods=2).max() - pd.Series(low).rolling(window=2, min_periods=2).min()
    tr2 = abs(pd.Series(high) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low) - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14 = highest_high_14 - lowest_low_14
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    
    chop = 100 * np.log10(pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values / np.log10(range_14))
    
    # Chop regimes: >61.8 = ranging (choppy), <38.2 = trending
    chop_ranging = chop > 61.8
    chop_trending = chop < 38.2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(chop_ranging[i]) or
            np.isnan(chop_trending[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR 1d EMA50 turns bearish
            if close[i] < lowest_20[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR 1d EMA50 turns bullish
            if close[i] > highest_20[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and regime filter
            # Long breakout: price above Donchian high + volume spike + (trending OR ranging at support)
            if (close[i] > highest_20[i] and 
                volume_spike[i] and 
                (chop_trending[i] or (chop_ranging[i] and close[i] > ema_50_aligned[i]))):
                position = 1
                signals[i] = 0.25
            # Short breakout: price below Donchian low + volume spike + (trending OR ranging at resistance)
            elif (close[i] < lowest_20[i] and 
                  volume_spike[i] and 
                  (chop_trending[i] or (chop_ranging[i] and close[i] < ema_50_aligned[i]))):
                position = -1
                signals[i] = -0.25
    
    return signals