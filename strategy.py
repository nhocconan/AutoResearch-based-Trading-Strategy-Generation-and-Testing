#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume spike and 1w choppiness regime filter
# - Long when price breaks above 4h Donchian(20) high AND 1d volume > 1.8x 20-period volume SMA AND 1w chop < 38.2 (trending)
# - Short when price breaks below 4h Donchian(20) low AND 1d volume > 1.8x 20-period volume SMA AND 1w chop < 38.2 (trending)
# - Exit: price retreats to 4h Donchian(20) midpoint or volume drops below average
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 19-50 trades/year on 4h timeframe to stay within fee drag limits
# - Uses Donchian channels for structure, volume for confirmation, chop regime to avoid whipsaws

name = "4h_1d_1w_donchian_volume_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 1w choppiness index
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range for 1w
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Choppy index: 100 * log10(sum(TR, n) / (n * (max(high,n) - min(low,n)))) / log10(n)
    n_chop = 14
    tr_sum = pd.Series(tr).rolling(window=n_chop, min_periods=n_chop).sum().values
    max_high = pd.Series(high_1w).rolling(window=n_chop, min_periods=n_chop).max().values
    min_low = pd.Series(low_1w).rolling(window=n_chop, min_periods=n_chop).min().values
    denominator = n_chop * (max_high - min_low)
    chop = 100 * np.log10(tr_sum / denominator) / np.log10(n_chop)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Align 1d volume confirmation to 4h timeframe
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    for i in range(50, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(donchian_mid[i]) or np.isnan(volume_sma_20_1d_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.8x 20-period volume SMA
        # Get current 1d volume (aligned to 4h)
        vol_1d_idx = i // 6  # 6 4h bars in 1d
        if vol_1d_idx < len(volume_1d):
            vol_1d_current = volume_1d[vol_1d_idx]
            vol_confirm = vol_1d_current > 1.8 * volume_sma_20_1d_aligned[i]
        else:
            vol_confirm = False
        
        # Regime filter: 1w chop < 38.2 (trending market)
        regime_filter = chop_aligned[i] < 38.2
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous low
        
        # Exit conditions: price retreats to midpoint or loss of volume/regime
        exit_long = close[i] < donchian_mid[i] or not (vol_confirm and regime_filter)
        exit_short = close[i] > donchian_mid[i] or not (vol_confirm and regime_filter)
        
        if position == 0:  # Flat - look for entry
            if breakout_up and vol_confirm and regime_filter:
                position = 1
                signals[i] = 0.25
            elif breakout_down and vol_confirm and regime_filter:
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