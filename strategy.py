#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and chop regime filter
# - Long when price breaks above Donchian(20) high AND volume > 1.5x 20-bar avg AND chop > 61.8 (range)
# - Short when price breaks below Donchian(20) low AND volume > 1.5x 20-bar avg AND chop > 61.8 (range)
# - Exit when price crosses Donchian(20) midpoint OR chop < 38.2 (trending)
# - Uses 1d HTF for Donchian calculation (proven pattern from DB)
# - Volume confirmation avoids low-liquidity false breakouts
# - Chop regime filter ensures we only trade breakouts from ranging markets
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Works in both bull and bear markets: chop filter adapts to regime, volume confirms institutional interest

name = "4h_1d_donchian_breakout_volume_chop_v1"
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
    
    # Pre-compute 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian high: rolling max of high
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian low: rolling min of low
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    # Donchian midpoint: average of high and low
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.5 * volume_20_avg)
    
    # Pre-compute 1d Chopiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR(1),14) / (log10(HH(14)-LL(14)) * sqrt(14)))
    tr1 = np.maximum(
        np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1])),
        np.abs(low_1d[1:] - close_1d[:-1])
    )
    # Pad TR to match length (first TR is undefined)
    tr1 = np.concatenate([[np.nan], tr1])
    atr1_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    hh14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero and log of zero/negative
    hl_range = hh14 - ll14
    chop_raw = 100 * np.log10(atr1_sum / (np.log10(hl_range) * np.sqrt(14)))
    # Replace invalid values with 50 (neutral)
    chop = np.where((hl_range <= 0) | (atr1_sum <= 0) | np.isnan(atr1_sum) | np.isnan(hl_range), 50, chop_raw)
    
    # Align HTF indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Pre-compute session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or np.isnan(vol_spike_aligned[i]) or
            np.isnan(chop_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Apply session filter
        if not in_session[i]:
            # Outside session: flatten position
            position = 0
            signals[i] = 0.0
            continue
        
        price = prices['close'].iloc[i]
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian high AND volume spike AND chop > 61.8 (range)
            if (price > donchian_high_aligned[i] and 
                vol_spike_aligned[i] and 
                chop_aligned[i] > 61.8):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian low AND volume spike AND chop > 61.8 (range)
            elif (price < donchian_low_aligned[i] and 
                  vol_spike_aligned[i] and 
                  chop_aligned[i] > 61.8):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price crosses Donchian midpoint OR chop < 38.2 (trending)
            exit_signal = (price > donchian_mid_aligned[i] and position == -1) or \
                          (price < donchian_mid_aligned[i] and position == 1) or \
                          (chop_aligned[i] < 38.2)
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals