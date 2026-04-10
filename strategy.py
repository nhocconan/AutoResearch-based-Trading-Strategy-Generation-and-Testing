#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme Reversion with 1d Volume Regime Filter
# - Primary: 6h timeframe balances trade frequency and fee drag
# - HTF: 1d for volume confirmation and regime filtering
# - Logic: Williams %R identifies overbought/oversold conditions; mean reversion when extreme readings reverse
# - Long: Williams %R(14) crosses above -90 from below + 1d volume > 1.5x 20-period MA
# - Short: Williams %R(14) crosses below -10 from above + 1d volume > 1.5x 20-period MA
# - Exit: Williams %R crosses -50 (mean reversion midpoint) or opposite extreme reached
# - Position sizing: 0.25 (discrete level)
# - Target: 80-180 total trades over 4 years (20-45/year) - within 6h sweet spot
# - Works in bull/bear: Mean reversion effective in ranging markets (2025), extremes catch reversals in trending markets

name = "6h_1d_williamsr_extreme_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 6h Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_6h) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        volume_spike = volume_1d[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Williams %R crossover signals (using previous bar to detect cross)
            if i > 0:
                # Long entry: Williams %R crosses above -90 from below (oversold reversal)
                long_signal = (williams_r[i-1] <= -90 and williams_r[i] > -90 and volume_spike)
                # Short entry: Williams %R crosses below -10 from above (overbought reversal)
                short_signal = (williams_r[i-1] >= -10 and williams_r[i] < -10 and volume_spike)
                
                if long_signal:
                    position = 1
                    signals[i] = 0.25
                elif short_signal:
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Williams %R crosses -50 (mean reversion midpoint)
            # 2. Williams %R reaches opposite extreme (overbought/oversold continuation)
            
            if i > 0:
                if position == 1:  # Long position
                    exit_condition = (
                        williams_r[i-1] >= -50 and williams_r[i] < -50 or  # Crossed below -50
                        williams_r[i] >= -10  # Reached overbought territory
                    )
                    if exit_condition:
                        position = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = 0.25
                else:  # position == -1 (Short position)
                    exit_condition = (
                        williams_r[i-1] <= -50 and williams_r[i] > -50 or  # Crossed above -50
                        williams_r[i] <= -90  # Reached oversold territory
                    )
                    if exit_condition:
                        position = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = -0.25
            else:
                signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
    
    return signals