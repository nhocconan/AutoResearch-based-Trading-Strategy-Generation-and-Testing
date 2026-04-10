#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and choppiness regime filter
# - Long when price breaks above Donchian(20) high AND 1d volume > 1.5x 20-bar avg AND chop < 61.8 (trending)
# - Short when price breaks below Donchian(20) low AND 1d volume > 1.5x 20-bar avg AND chop < 61.8 (trending)
# - Exit when price touches Donchian(20) midpoint OR chop > 61.8 (range) to avoid whipsaw
# - Uses discrete position sizing (0.25) to balance return and drawdown
# - Donchian captures structural breaks; volume confirmation avoids low-liquidity false signals
# - Choppiness filter (CHOP < 61.8) ensures we only trade in trending markets, avoiding range-bound losses
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets: breaks trends persist; chop filter avoids false breakouts in ranges

name = "12h_1d_donchian_volume_chop_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * volume_20_avg_1d)
    
    # Pre-compute 1d choppiness index: CHOP(14) = 100 * log10(sum(ATR(14)) / log10(range(14))) / log10(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], 0])], np.maximum(np.maximum(tr1, tr2), tr3)])
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_14 - ll_14
    chop = np.where(range_14 != 0, 100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 50)
    chop = np.where(np.isnan(chop), 50, chop)
    chop_trending = chop < 61.8  # Trending regime
    
    # Align HTF indicators to 12h timeframe
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    chop_trending_aligned = align_htf_to_ltf(prices, df_1d, chop_trending)
    
    # Pre-compute Donchian channels on 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Donchian breakout conditions
    donchian_breakout_up = high > donchian_high  # Break above upper band
    donchian_breakout_down = low < donchian_low   # Break below lower band
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(donchian_window, n):  # Start after Donchian warmup
        # Skip if any required data is invalid
        if (np.isnan(vol_spike_1d_aligned[i]) or np.isnan(chop_trending_aligned[i]) or
            np.isnan(donchian_breakout_up[i]) or np.isnan(donchian_breakout_down[i]) or
            np.isnan(donchian_mid[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when Donchian breakout up AND volume spike AND trending regime
            if (donchian_breakout_up[i] and 
                vol_spike_1d_aligned[i] and 
                chop_trending_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when Donchian breakout down AND volume spike AND trending regime
            elif (donchian_breakout_down[i] and 
                  vol_spike_1d_aligned[i] and 
                  chop_trending_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit conditions
            # Exit when price touches Donchian midpoint OR chop > 61.8 (range)
            exit_signal = (np.abs(close[i] - donchian_mid[i]) < 0.001 * close[i]) or (not chop_trending_aligned[i])
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals