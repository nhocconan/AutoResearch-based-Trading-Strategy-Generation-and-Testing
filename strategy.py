#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d volume confirmation and choppiness regime filter
# - Long when Williams %R < -80 (oversold) + volume > 1.5x 20-period 1d volume SMA + CHOP(14) > 61.8 (ranging market for mean reversion)
# - Short when Williams %R > -20 (overbought) + volume > 1.5x 20-period 1d volume SMA + CHOP(14) > 61.8 (ranging market for mean reversion)
# - Exit: Williams %R returns to -50 (mean reversion midpoint)
# - Position sizing: 0.25 discrete level
# - Williams %R identifies overextended moves in ranging markets
# - Volume confirmation ensures institutional participation
# - Choppiness filter ensures we only trade in ranging markets to avoid trend-following false signals
# - Works in bull/bear: ranging periods occur in all regimes, providing mean reversion opportunities

name = "4h_1d_williamsr_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Calculate Williams %R on primary timeframe (4h)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        (highest_high - close) / (highest_high - lowest_low) * -100,
        -50  # default to neutral when range is zero
    )
    
    # Calculate 1d volume SMA for confirmation (20-period)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate Choppiness Index on 1d timeframe (14-period)
    # CHOP = 100 * log10(sum(ATR)/ (n * (max(high)-min(low)))) / log10(n)
    # Lower CHOP = trending, Higher CHOP = ranging
    # We want CHOP > 61.8 for ranging markets (mean reversion)
    
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = df_1d['high'].iloc[0] - df_1d['low'].iloc[0]  # first bar
    
    # Sum of ATR over 14 periods
    atr_sum_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Max high and min low over 14 periods
    max_high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop_1d = np.where(
        (max_high_14 - min_low_14) != 0,
        100 * np.log10(atr_sum_14 / (14 * (max_high_14 - min_low_14))) / np.log10(14),
        50  # default to neutral when range is zero
    )
    
    # Align Choppiness Index to 4h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume for confirmation (aligned to 4h)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm = vol_1d_current[i] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Regime filter: CHOP > 61.8 indicates ranging market (higher = more ranging)
        regime_filter = chop_1d_aligned[i] > 61.8
        
        # Williams %R mean reversion entry conditions
        # Long: Williams %R < -80 (oversold) + volume confirmation + ranging regime
        # Short: Williams %R > -20 (overbought) + volume confirmation + ranging regime
        long_entry = (williams_r[i] < -80 and 
                     vol_confirm and 
                     regime_filter)
        short_entry = (williams_r[i] > -20 and 
                      vol_confirm and 
                      regime_filter)
        
        # Exit conditions: Williams %R returns to -50 (mean reversion midpoint)
        exit_long = williams_r[i] > -50
        exit_short = williams_r[i] < -50
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
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