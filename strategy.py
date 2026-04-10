#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d volume confirmation and trend filter
# - Primary signal: Bollinger Band width at 20-period low + price breaks above/below bands
# - Volume filter: 1d volume > 1.3x 30-period average volume (institutional participation)
# - Trend filter: Price > 1d EMA50 for longs, Price < 1d EMA50 for shorts (avoid counter-trend)
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: Opposite Bollinger Band (long exits at lower band, short exits at upper band)
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Squeeze breakouts capture volatility expansion; filters avoid false signals

name = "6h_1d_bb_squeeze_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d volume spike filter
    volume_1d = df_1d['volume'].values
    avg_volume_30 = pd.Series(volume_1d).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume_1d > (1.3 * avg_volume_30)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Pre-compute 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Pre-compute 6h Bollinger Bands (20, 2.0)
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    
    sma_20 = pd.Series(close_6h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_6h).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + (2.0 * std_20)
    bb_lower = sma_20 - (2.0 * std_20)
    bb_width = (bb_upper - bb_lower) / sma_20  # Normalized width
    
    # Bollinger Band squeeze: width at 20-period low
    bb_width_min = pd.Series(bb_width).rolling(window=20, min_periods=20).min().values
    bb_squeeze = bb_width <= bb_width_min
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or
            np.isnan(vol_spike_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(bb_squeeze[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Bollinger middle band (mean reversion)
            if close_6h[i] < sma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Bollinger middle band (mean reversion)
            if close_6h[i] > sma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Bollinger Band squeeze breakouts with volume and trend filters
            if bb_squeeze[i] and vol_spike_aligned[i]:
                # Long: price breaks above upper band AND above 1d EMA50 (uptrend)
                if close_6h[i] > bb_upper[i] and close_6h[i] > ema_50_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below lower band AND below 1d EMA50 (downtrend)
                elif close_6h[i] < bb_lower[i] and close_6h[i] < ema_50_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals