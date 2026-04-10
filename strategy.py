#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d regime filter
# - Bull Power = High - EMA13(close); Bear Power = EMA13(close) - Low
# - Long when Bull Power > 0 AND Bear Power < 0 AND 1d close > 1d SMA50 (bull regime)
# - Short when Bear Power > 0 AND Bull Power < 0 AND 1d close < 1d SMA50 (bear regime)
# - Exit when power signs diverge or regime flips
# - Uses 1d regime to avoid counter-trend trades in strong trends
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Position size: 0.25 discrete to minimize fee churn
# - Works in both bull and bear by adapting to 1d regime

name = "6h_1d_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # 1d regime filter: SMA50
    close_1d = df_1d['close'].values
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Regime: bullish if close > SMA50, bearish if close < SMA50
    regime_bullish = close_1d_aligned > sma_50_1d_aligned
    regime_bearish = close_1d_aligned < sma_50_1d_aligned
    
    for i in range(50, n):  # Start after warmup for EMA13 and SMA50
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(sma_50_1d_aligned[i]) or np.isnan(close_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Elder Ray signals with regime filter
        long_signal = bull_power[i] > 0 and bear_power[i] < 0 and regime_bullish[i]
        short_signal = bear_power[i] > 0 and bull_power[i] < 0 and regime_bearish[i]
        
        # Exit conditions: power divergence or regime flip
        exit_long = (bull_power[i] <= 0 or bear_power[i] >= 0 or not regime_bullish[i])
        exit_short = (bear_power[i] <= 0 or bull_power[i] >= 0 or not regime_bearish[i])
        
        if position == 0:  # Flat - look for entry
            if long_signal:
                position = 1
                signals[i] = 0.25
            elif short_signal:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals