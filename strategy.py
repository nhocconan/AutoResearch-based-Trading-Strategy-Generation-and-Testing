#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian Channel Breakout with 1w volume regime filter
# - Primary: 12h timeframe for lower frequency and reduced fee drag
# - HTF: 1w for volatility regime (volume-based) and trend confirmation
# - Long: Price breaks above 20-period Donchian high + 1w volume > 1.5x 4-week MA
# - Short: Price breaks below 20-period Donchian low + 1w volume > 1.5x 4-week MA
# - Exit: Price reverts to 20-period Donchian midpoint (mean reversion)
# - Position sizing: 0.25 (discrete level)
# - Target: 80-150 total trades over 4 years (20-38/year) - within 12h sweet spot
# - Works in bull/bear: Donchian breakouts capture trends, mean reversion exit works in ranging markets (2025)

name = "12h_1w_donchian_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute 12h OHLCV
    open_12h = prices['open'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    volume_12h = prices['volume'].values
    
    # Pre-compute 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 20-period Donchian Channel on 12h
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate 1w volume moving average (4-period) for regime filter
    volume_ma_4_1w = pd.Series(volume_1w).rolling(window=4, min_periods=4).mean().values
    volume_ma_4_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_4_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma_4_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # 1w volume regime: current week volume > 1.5x 4-week MA (avoid low-vol chop)
        volume_regime = volume_1w[i // (7*24*4)] > 1.5 * volume_ma_4_1w_aligned[i] if i // (7*24*4) < len(volume_1w) else False
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian high + volume regime
            if close_12h[i] > donchian_high[i] and volume_regime:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low + volume regime
            elif close_12h[i] < donchian_low[i] and volume_regime:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit condition: Price reverts to Donchian midpoint (mean reversion)
            if position == 1:  # Long position
                if close_12h[i] < donchian_mid[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close_12h[i] > donchian_mid[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals