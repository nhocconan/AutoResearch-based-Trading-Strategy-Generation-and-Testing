#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike filter and chop regime
# - Long when price breaks above Donchian(20) high AND 1d volume > 1.5x 20-period volume SMA AND chop(14) > 61.8 (range regime)
# - Short when price breaks below Donchian(20) low AND 1d volume > 1.5x 20-period volume SMA AND chop(14) > 61.8 (range regime)
# - Exit: price retraces to Donchian midpoint OR chop < 38.2 (trend regime) OR volume drops below average
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 75-200 total trades over 4 years (19-50/year) within fee drag limits
# - Uses Donchian channels for structure, volume spike for confirmation, chop filter for regime

name = "4h_1d_donchian_volume_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 4h Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 4h chop regime (choppiness index)
    # Chop = 100 * log10(sum(ATR(1), n) / (log10(n) * (max(high,n) - min(low,n))))
    tr1 = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr1 = np.concatenate([[np.nan], tr1])  # align with original arrays
    atr1 = pd.Series(tr1).rolling(window=1, min_periods=1).mean().values  # ATR(1) = TR
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr1 / (np.log10(14) * (max_high - min_low)))
    chop = np.where((max_high - min_low) == 0, 50, chop)  # avoid division by zero
    
    for i in range(40, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.5x 20-period volume SMA
        # Need to get 1d volume for current aligned bar
        idx_1d = i // 6  # 4h bars per 1d (24h/4h = 6)
        if idx_1d >= len(volume_1d):
            vol_confirm = False
        else:
            vol_confirm = volume_1d[idx_1d] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Chop regime filter: chop > 61.8 indicates range-bound market (good for mean reversion/breakouts in range)
        chop_filter = chop[i] > 61.8
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous Donchian high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous Donchian low
        
        # Exit conditions: price retraces to midpoint OR chop < 38.2 (trend regime) OR volume drops below average
        exit_long = close[i] < donchian_mid[i] or chop[i] < 38.2 or not vol_confirm
        exit_short = close[i] > donchian_mid[i] or chop[i] < 38.2 or not vol_confirm
        
        if position == 0:  # Flat - look for entry
            if breakout_up and vol_confirm and chop_filter:
                position = 1
                signals[i] = 0.25
            elif breakout_down and vol_confirm and chop_filter:
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