#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + 1d choppiness regime filter
# - Long: price breaks above Donchian(20) high AND 1d volume > 1.5x 20-period average AND 1d chop > 61.8 (range regime)
# - Short: price breaks below Donchian(20) low AND 1d volume > 1.5x 20-period average AND 1d chop < 38.2 (trend regime)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss (2.0x ATR(14)) to manage risk
# - Designed for 4h timeframe: targets 19-50 trades/year to avoid fee drag
# - Works in bull/bear markets: chop regime filter adapts to market conditions, volume confirms breakout validity

name = "4h_1d_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Donchian high: rolling max of high over 20 periods
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Donchian low: rolling min of low over 20 periods
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 1d volume spike
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.5 * avg_volume_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Pre-compute 1d choppiness index
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop = 100 * log15(sum(ATR14) / (max(high) - min(low)) over 14 periods)
    sum_atr = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr / (max_high - min_low + 1e-10)) / np.log10(15)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Pre-compute 4h ATR(14) for stoploss
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = tr1_4h[0]
    atr_14_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_spike_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(atr_14_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR stoploss hit
            if close_4h[i] < donch_low[i] or close_4h[i] < entry_price - 2.0 * atr_14_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR stoploss hit
            if close_4h[i] > donch_high[i] or close_4h[i] > entry_price + 2.0 * atr_14_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume and chop filters
            if vol_spike_aligned[i]:
                # Long: price breaks above Donchian high AND chop > 61.8 (range regime)
                if close_4h[i] > donch_high[i] and chop_aligned[i] > 61.8:
                    position = 1
                    entry_price = close_4h[i]
                    signals[i] = 0.25
                # Short: price breaks below Donchian low AND chop < 38.2 (trend regime)
                elif close_4h[i] < donch_low[i] and chop_aligned[i] < 38.2:
                    position = -1
                    entry_price = close_4h[i]
                    signals[i] = -0.25
    
    return signals