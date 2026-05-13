#!/usr/bin/env python3
# Hypothesis: 6h Bollinger Band Width Regime + 1d Donchian(20) Breakout
# Uses Bollinger Band Width (BBW) percentile to detect low volatility squeeze regimes.
# In low volatility (BBW < 20th percentile), waits for 1d Donchian breakout for entry.
# In high volatility (BBW > 80th percentile), fades extreme price moves at Bollinger Bands.
# Combines volatility regime filter with multi-timeframe structure for 12-35 trades/year.
# Works in bull/bear/range markets by adapting to volatility conditions.

name = "6h_BBW_Regime_1dDonchian_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Bollinger Bands (20, 2) on 6h close
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    bb_width = (upper_bb - lower_bb) / sma20  # Normalized BB Width
    
    # BB Width percentile rank (50-period lookback) for regime detection
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Get 1d data for HTF Donchian breakout
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Donchian channels (20-period)
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(sma20[i]) or np.isnan(std20[i]) or np.isnan(bb_width_percentile[i]) or \
           np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]):
            signals[i] = 0.0
            continue
        
        bbwp = bb_width_percentile[i]
        
        if position == 0:
            # ENTRY CONDITIONS
            # Low volatility regime (squeeze): BBW < 20th percentile
            if bbwp < 20:
                # Wait for 1d Donchian breakout in direction of 6h trend (price > SMA20)
                if close[i] > donch_high_20_aligned[i] and close[i] > sma20[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donch_low_20_aligned[i] and close[i] < sma20[i]:
                    signals[i] = -0.25
                    position = -1
            # High volatility regime (expansion): BBW > 80th percentile
            elif bbwp > 80:
                # Fade extreme moves at Bollinger Bands
                if close[i] >= upper_bb[i]:
                    signals[i] = -0.25  # Short at upper band
                    position = -1
                elif close[i] <= lower_bb[i]:
                    signals[i] = 0.25   # Long at lower band
                    position = 1
        
        elif position == 1:
            # EXIT LONG: BBW expansion signal or mean reversion in high volatility
            if bbwp > 80 and close[i] <= sma20[i]:  # Mean reversion signal
                signals[i] = 0.0
                position = 0
            elif close[i] < lower_bb[i]:  # Stop at lower band
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # EXIT SHORT: BBW expansion signal or mean reversion in high volatility
            if bbwp > 80 and close[i] >= sma20[i]:  # Mean reversion signal
                signals[i] = 0.0
                position = 0
            elif close[i] > upper_bb[i]:  # Stop at upper band
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals