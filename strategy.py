#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray power with 1d regime filter and volume confirmation
# - Bull Power = High - EMA13(1d), Bear Power = EMA13(1d) - Low
# - Long when Bull Power > 0 AND Bear Power < 0 AND 1d volume > 1.5x 20-bar avg AND 1d close > 1d open
# - Short when Bear Power > 0 AND Bull Power < 0 AND 1d volume > 1.5x 20-bar avg AND 1d close < 1d open
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Elder Ray measures bull/bear strength relative to 1d EMA; regime filter ensures alignment with daily trend
# - Volume confirmation adds conviction to signals, reducing false breakouts

name = "6h_1d_elder_ray_power_regime_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d EMA13 for Elder Ray power calculation
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute Elder Ray power components: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = ema13_1d - low_1d
    
    # Align 1d Elder Ray power to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Pre-compute 1d regime filter: bullish if close > open, bearish if close < open
    open_1d = df_1d['open'].values
    close_1d = df_1d['close'].values
    daily_bullish = close_1d > open_1d
    daily_bearish = close_1d < open_1d
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish)
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * volume_20_avg)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(daily_bullish_aligned[i]) or np.isnan(daily_bearish_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long when Bull Power > 0 AND Bear Power < 0 (bulls in control) AND volume spike AND daily bullish
            if (bull_power_aligned[i] > 0 and 
                bear_power_aligned[i] < 0 and 
                vol_spike_1d_aligned[i] and 
                daily_bullish_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when Bear Power > 0 AND Bull Power < 0 (bears in control) AND volume spike AND daily bearish
            elif (bear_power_aligned[i] > 0 and 
                  bull_power_aligned[i] < 0 and 
                  vol_spike_1d_aligned[i] and 
                  daily_bearish_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit when power weakens
            # Exit when Bull Power <= 0 (for long) or Bear Power <= 0 (for short)
            exit_signal = False
            if position == 1:  # Long position
                if bull_power_aligned[i] <= 0:
                    exit_signal = True
            elif position == -1:  # Short position
                if bear_power_aligned[i] <= 0:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals