#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and chop regime filter
# - Long when price breaks above Donchian(20) high AND 1d volume > 1.5x 20-bar avg AND chop < 61.8 (trending)
# - Short when price breaks below Donchian(20) low AND 1d volume > 1.5x 20-bar avg AND chop < 61.8 (trending)
# - Exit when price touches Donchian(20) midpoint OR chop > 61.8 (range)
# - Uses discrete position sizing (0.25) to balance return and drawdown
# - Donchian captures breakouts; volume confirmation avoids low-liquidity false signals
# - Chop filter (14) avoids whipsaws in ranging markets (CHOP > 61.8 = range)
# - Works in both bull and bear markets: breakouts occur in all regimes, chop filter improves quality
# - Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe

name = "4h_1d_donchian_breakout_volume_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period) on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian high: highest high over past 20 bars
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian low: lowest low over past 20 bars
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Donchian midpoint: average of high and low
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * volume_20_avg_1d)
    
    # Pre-compute Choppiness Index (14) on 1d data
    # CHOP = 100 * log10(sum(ATR over n) / (log10(n) * (highest high - lowest low)))
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    highest_high = df_1d['high'].rolling(window=14, min_periods=14).max()
    lowest_low = df_1d['low'].rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr.rolling(window=14, min_periods=14).sum() / 
                          (np.log10(14) * (highest_high - lowest_low)))
    chop_values = chop.values
    chop_trending = chop_values < 61.8  # CHOP < 61.8 = trending (good for breakouts)
    chop_ranging = chop_values > 61.8   # CHOP > 61.8 = ranging (avoid breakouts)
    
    # Align HTF indicators to 4h timeframe
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    chop_trending_aligned = align_htf_to_ltf(prices, df_1d, chop_trending)
    chop_ranging_aligned = align_htf_to_ltf(prices, df_1d, chop_ranging)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(vol_spike_1d_aligned[i]) or
            np.isnan(chop_trending_aligned[i]) or np.isnan(chop_ranging_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian high AND volume spike AND trending regime
            if (close[i] > donchian_high[i] and 
                vol_spike_1d_aligned[i] and 
                chop_trending_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian low AND volume spike AND trending regime
            elif (close[i] < donchian_low[i] and 
                  vol_spike_1d_aligned[i] and 
                  chop_trending_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit conditions
            # Exit when price touches Donchian midpoint OR market becomes ranging
            exit_signal = (abs(close[i] - donchian_mid[i]) < 0.001 * close[i]) or chop_ranging_aligned[i]
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals