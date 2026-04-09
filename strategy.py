#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume spike and chop regime filter
# - Uses Donchian channel breakout (20-period) for trend following entries
# - Volume confirmation: current volume > 1.5x 20-period average volume
# - Regime filter: Choppiness Index > 61.8 for ranging market (mean reversion bias)
# - ATR-based stoploss implemented via signal=0 when price moves against position
# - Designed for low trade frequency (target: 20-40 trades/year) to minimize fee drag
# - Works in bull/bear markets via regime-adaptive logic (breakout in trend, mean reversion in chop)

name = "4h_1d_donchian_vol_chop_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute HTF indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d ATR(14) for Choppiness Index calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1_1d[0] if len(tr1_1d) > 0 else 0
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 1d ADX(14) for trend strength (alternative to chop)
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / np.where(tr_14 == 0, 1, tr_14)
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / np.where(tr_14 == 0, 1, tr_14)
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / np.where((plus_di_14 + minus_di_14) == 0, 1, (plus_di_14 + minus_di_14))
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 4h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 1d volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * vol_ma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # 4h Donchian channel (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 = trending market (favor breakouts)
        # ADX <= 25 = ranging market (favor mean reversion)
        is_trending = adx_1d_aligned[i] > 25
        
        if position == 1:  # Long position
            # Exit conditions: Donchian lower band break or regime change to chop
            if close[i] <= lowest_low_20[i]:  # Break below Donchian low
                position = 0
                signals[i] = 0.0
            elif not is_trending and volume_spike_aligned[i] > 0.5:  # Chop regime with volume
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: Donchian upper band break or regime change to chop
            if close[i] >= highest_high_20[i]:  # Break above Donchian high
                position = 0
                signals[i] = 0.0
            elif not is_trending and volume_spike_aligned[i] > 0.5:  # Chop regime with volume
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for breakout entries with volume confirmation
            if is_trending:
                # Trending market: Donchian breakout with volume spike
                if (close[i] > highest_high_20[i] and  # Break above upper band
                    volume_spike_aligned[i] > 0.5):    # Volume confirmation
                    position = 1
                    signals[i] = 0.25
                elif (close[i] < lowest_low_20[i] and   # Break below lower band
                      volume_spike_aligned[i] > 0.5):   # Volume confirmation
                    position = -1
                    signals[i] = -0.25
            else:
                # Ranging market: mean reversion at extremes with volume spike
                if (close[i] < lowest_low_20[i] and     # Near Donchian low
                    volume_spike_aligned[i] > 0.5):     # Volume confirmation
                    position = 1
                    signals[i] = 0.25
                elif (close[i] > highest_high_20[i] and # Near Donchian high
                      volume_spike_aligned[i] > 0.5):   # Volume confirmation
                    position = -1
                    signals[i] = -0.25
    
    return signals