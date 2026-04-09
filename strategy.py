#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w HTF Donchian channel breakout with volume confirmation
# - Uses 1w HTF for Donchian(20) upper/lower channels (based on completed weekly candles)
# - Long when price breaks above Donchian upper channel with volume > 2.0x 20-period average
# - Short when price breaks below Donchian lower channel with volume > 2.0x 20-period average
# - Fixed position size 0.25 to control drawdown
# - Exit on opposite Donchian breakout (reversal system)
# - Works in bull/bear: Donchian channels adapt to volatility, volume confirmation filters false breakouts
# - Target: 15-30 trades/year on 1d timeframe (60-120 total over 4 years)

name = "1d_1w_donchian_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w Donchian channel (20-period) - based on completed weekly candles
    high_max_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (wait for completed 1w bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, high_max_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, low_min_20)
    
    # Pre-compute volume confirmation (20-period average for 1d)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x average
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit long: price breaks below Donchian lower channel with volume confirmation
            if volume_confirmed and close[i] < donchian_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit short: price breaks above Donchian upper channel with volume confirmation
            if volume_confirmed and close[i] > donchian_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Donchian breakout with volume confirmation
            if volume_confirmed:
                # Long entry: price breaks above Donchian upper channel
                if close[i] > donchian_high_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below Donchian lower channel
                elif close[i] < donchian_low_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals