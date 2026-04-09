#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian breakout with volume confirmation and chop regime filter
# - Uses 1d HTF for Donchian(20) breakout: price > 20-period high = bullish, < 20-period low = bearish
# - Volume confirmation: current 12h volume > 1.5x 20-period average to avoid low-volume false signals
# - Chop regime filter: only trade when choppiness index > 61.8 (ranging market) for mean reversion at extremes
# - In ranging markets: fade Donchian breakouts (short at upper band, long at lower band)
# - In trending markets (chop < 38.2): don't trade (avoid whipsaws)
# - Fixed position size 0.25 to control drawdown
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_donchian_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20 periods)
    period20_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align HTF data to 12h timeframe (wait for completed 1d bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, period20_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, period20_low)
    
    # Pre-compute volume confirmation (20-period average for 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute choppiness index (14 periods) on 12h timeframe
    # Chop = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(n)
    atr_values = np.zeros(n)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with indices
    
    atr_ma = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop_denominator = highest_high - lowest_low
    chop = np.where(chop_denominator > 0, 
                    100 * np.log10(pd.Series(atr_ma).rolling(window=14, min_periods=14).sum().values / chop_denominator) / np.log10(14),
                    50)  # neutral when denominator is 0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(chop[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Chop regime: > 61.8 = ranging (mean revert), < 38.2 = trending (don't trade)
        ranging_market = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit conditions: price reaches Donchian low or volume fails
            if close[i] <= donchian_low_aligned[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit conditions: price reaches Donchian high or volume fails
            if close[i] >= donchian_high_aligned[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Entry logic: only in ranging markets with volume confirmation
            if volume_confirmed and ranging_market:
                if close[i] >= donchian_high_aligned[i]:
                    # Price at upper Donchian band: short mean reversion
                    position = -1
                    signals[i] = -position_size
                elif close[i] <= donchian_low_aligned[i]:
                    # Price at lower Donchian band: long mean reversion
                    position = 1
                    signals[i] = position_size
    
    return signals