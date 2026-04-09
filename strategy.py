#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d ATR-based volatility regime filter + volume spike + Donchian(20) breakout
# In low volatility regime (ATR contraction): use Donchian breakout for trend continuation
# In high volatility regime (ATR expansion): use mean reversion at Donchian channels
# Volume confirmation required for all entries to avoid false breakouts
# Designed to work in both bull (trend following) and bear (mean reversion in high vol) markets
# Discrete position sizing 0.25 targets ~30-60 trades/year to minimize fee drag

name = "4h_1d_atr_regime_donchian_v1"
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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.zeros_like(close_1d)
    
    # Calculate 1d ATR(20) for volatility regime
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
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
    
    atr_1d = wilders_smoothing(tr, 20)
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    
    # Volatility regime: low vol (trending) when ATR < MA, high vol (mean revert) when ATR > MA
    vol_regime = np.where(atr_ma_1d > 0, atr_1d / atr_ma_1d, 1.0)
    low_vol = vol_regime < 0.8   # ATR contraction = trending regime
    high_vol = vol_regime > 1.2  # ATR expansion = mean reversion regime
    
    # Calculate 4h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h average volume (20-period) for volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    low_vol_aligned = align_htf_to_ltf(prices, df_1d, low_vol)
    high_vol_aligned = align_htf_to_ltf(prices, df_1d, high_vol)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(low_vol_aligned[i]) or 
            np.isnan(high_vol_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        if position == 1:  # Long position
            if low_vol_aligned[i] and volume_confirmed:
                # Exit long if price falls below Donchian low (trend break)
                if close[i] < lowest_20[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif high_vol_aligned[i]:
                # Exit long if price moves back above Donchian low (mean reversion exit)
                if close[i] > lowest_20[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if low_vol_aligned[i] and volume_confirmed:
                # Exit short if price rises above Donchian high (trend break)
                if close[i] > highest_20[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif high_vol_aligned[i]:
                # Exit short if price moves back below Donchian high (mean reversion exit)
                if close[i] < highest_20[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if low_vol_aligned[i] and volume_confirmed:
                # Breakout strategy in low volatility (trending) regime
                if close[i] > highest_20[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < lowest_20[i]:
                    position = -1
                    signals[i] = -0.25
            elif high_vol_aligned[i]:
                # Mean reversion at extremes in high volatility regime
                if close[i] < lowest_20[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] > highest_20[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals