#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h ATR-based volatility regime filter with Donchian(20) breakout
# In low volatility regime (ATR ratio < 0.8), use Donchian breakout for trend following
# In high volatility regime (ATR ratio > 1.2), use mean reversion at Donchian channels
# Volume confirmation required for breakouts to avoid false signals
# Discrete position sizing 0.25 to target ~30-60 trades/year and minimize fee drag
# Works in bull/bear markets: breakout captures trends in low vol, mean reversion profits from reversals in high vol

name = "4h_12h_atr_regime_donchian_v1"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h ATR(20) for volatility regime
    tr1 = np.abs(high_12h[1:] - low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_20_12h = wilders_smoothing(tr_12h, 20)
    atr_50_12h = wilders_smoothing(tr_12h, 50)
    
    # ATR ratio: short-term / long-term volatility
    atr_ratio = np.where(atr_50_12h != 0, atr_20_12h / atr_50_12h, 1.0)
    
    # Calculate 4h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2
    
    # Calculate 4h average volume (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h indicators to 4h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_12h, atr_ratio)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit conditions
            if atr_ratio_aligned[i] < 0.8:  # Low vol regime - trend following
                if close[i] < donchian_mid[i]:  # Exit at midpoint
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif atr_ratio_aligned[i] > 1.2:  # High vol regime - mean reversion
                if close[i] > lowest_20[i]:  # Exit when price returns above lower band
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Neutral regime - hold
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            if atr_ratio_aligned[i] < 0.8:  # Low vol regime - trend following
                if close[i] > donchian_mid[i]:  # Exit at midpoint
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif atr_ratio_aligned[i] > 1.2:  # High vol regime - mean reversion
                if close[i] < highest_20[i]:  # Exit when price returns below upper band
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Neutral regime - hold
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions
            if atr_ratio_aligned[i] < 0.8 and volume_confirmed:  # Low vol regime - breakout
                if close[i] > highest_20[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < lowest_20[i]:
                    position = -1
                    signals[i] = -0.25
            elif atr_ratio_aligned[i] > 1.2:  # High vol regime - mean reversion
                if close[i] < lowest_20[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] > highest_20[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals