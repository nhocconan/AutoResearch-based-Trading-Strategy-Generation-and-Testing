#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and chop regime filter
# - Primary: 4h timeframe (proven sweet spot for balance of signal quality and trade frequency)
# - HTF: 1d for volume confirmation (spike > 2.0x 20-period MA) and regime filter (chop > 61.8 = range)
# - Long: Price breaks above Donchian(20) high + 1d volume spike + chop > 61.8
# - Short: Price breaks below Donchian(20) low + 1d volume spike + chop > 61.8
# - Exit: Price returns to Donchian(20) midpoint OR chop < 38.2 (trending regime)
# - Position sizing: 0.30 (discrete level)
# - Works in bull/bear: Volume spikes confirm institutional interest; chop filter avoids false breakouts in strong trends
# - Target trades: ~100 total over 4 years (25/year) - well within fee-efficient range

name = "4h_1d_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h OHLCV
    open_4h = prices['open'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    # Upper band: 20-period high
    # Lower band: 20-period low
    # Middle band: (upper + lower) / 2
    high_roll_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_roll_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll_20
    donchian_lower = low_roll_20
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate 1d Choppiness Index (CHOP) - measures ranging vs trending markets
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(highest_high - lowest_low) * log10(period)))
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14 = highest_high_14 - lowest_low_14
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    
    # Chop calculation: CHOP = 100 * log10(sum(ATR) / (log10(range) * log10(period)))
    # Using common simplification: CHOP = 100 * log10(sum(ATR(14)) / log10(range)) / log10(14)
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    chop = 100 * (np.log10(sum_atr_14) / (np.log10(range_14) * np.log10(14)))
    # Handle invalid values
    chop = np.where(np.isnan(chop) | np.isinf(chop), 50.0, chop)
    
    # Align chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Get current 1d volume aligned to 4h
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime condition: chop > 61.8 (ranging market - good for mean reversion breakouts)
        ranging_regime = chop_aligned[i] > 61.8
        
        # Volume confirmation: current 1d volume > 2.0x 20-period MA (institutional participation)
        volume_spike = volume_1d_aligned[i] > 2.0 * volume_ma_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian upper + ranging regime + volume spike
            if (close_4h[i] > donchian_upper[i] and ranging_regime and volume_spike):
                position = 1
                signals[i] = 0.30
            # Short entry: Price breaks below Donchian lower + ranging regime + volume spike
            elif (close_4h[i] < donchian_lower[i] and ranging_regime and volume_spike):
                position = -1
                signals[i] = -0.30
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price returns to Donchian middle (mean reversion complete)
            # 2. Chop falls below 38.2 (trending regime - breakout likely to continue)
            
            if position == 1:  # Long position
                exit_condition = (
                    close_4h[i] < donchian_middle[i] or  # Price returned to middle
                    chop_aligned[i] < 38.2  # Trending regime
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.30
            else:  # position == -1 (Short position)
                exit_condition = (
                    close_4h[i] > donchian_middle[i] or  # Price returned to middle
                    chop_aligned[i] < 38.2  # Trending regime
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.30
    
    return signals