#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and ATR-based volatility filter
# In trending regimes (ATR(30) > ATR(60)*1.2): breakout above/below Donchian levels with volume confirmation
# In low volatility regimes (ATR(30) <= ATR(60)*1.2): no trades to avoid whipsaws
# Uses discrete position sizing 0.25 to limit trades to ~20-50/year and reduce fee drag
# Works in bull/bear markets: breakout catches strong moves, volatility filter avoids chop

name = "4h_1d_donchian_breakout_volume_volfilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ATR(30) and ATR(60) for volatility regime filter
    def calculate_atr(high, low, close, period):
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        def wilders_smoothing(values, period):
            if len(values) < period:
                return np.full(len(values), np.nan)
            alpha = 1.0 / period
            result = np.full(len(values), np.nan)
            result[period-1] = np.nanmean(values[:period])
            for i in range(period, len(values)):
                result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
            return result
        
        return wilders_smoothing(tr, period)
    
    atr_30_1d = calculate_atr(high_1d, low_1d, close_1d, 30)
    atr_60_1d = calculate_atr(high_1d, low_1d, close_1d, 60)
    vol_regime = atr_30_1d > (atr_60_1d * 1.2)  # Trending regime when short ATR > long ATR * 1.2
    
    # Calculate 1d average volume (20-period)
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * avg_volume_1d)  # Volume spike when > 2x average
    
    # Calculate 4h Donchian channels (20-period)
    def donchian_channels(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_20_4h_upper, donchian_20_4h_lower = donchian_channels(high, low, 20)
    
    # Align 1d indicators to 4h timeframe
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(vol_regime_aligned[i]) or np.isnan(volume_spike_aligned[i]) or
            np.isnan(donchian_20_4h_upper[i]) or np.isnan(donchian_20_4h_lower[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in trending volatility regime
        if not vol_regime_aligned[i]:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if price breaks below Donchian lower or volume spike disappears
            if close[i] < donchian_20_4h_lower[i] or not volume_spike_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price breaks above Donchian upper or volume spike disappears
            if close[i] > donchian_20_4h_upper[i] or not volume_spike_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long on breakout above Donchian upper with volume spike
            if close[i] > donchian_20_4h_upper[i] and volume_spike_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short on breakout below Donchian lower with volume spike
            elif close[i] < donchian_20_4h_lower[i] and volume_spike_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals