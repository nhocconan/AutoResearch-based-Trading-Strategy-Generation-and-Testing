#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + volume spike + choppiness regime filter.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- Regime filter: Choppiness Index(14) > 61.8 for ranging markets (mean reversion to Donchian middle).
- Entry: Long when close breaks above Donchian upper(20) AND volume > 2.0 * 4h volume MA(20) AND chop > 61.8;
         Short when close breaks below Donchian lower(20) AND volume > 2.0 * 4h volume MA(20) AND chop > 61.8.
- Exit: Close crosses Donchian middle(20) or ATR(14) stoploss (2.0 * ATR).
- Signal size: 0.25 discrete to control fee drag.
- Uses Donchian for structure, volume for participation, chop filter to avoid trending markets where breakouts fail.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channel (prior 20 periods)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian channel from prior 4h OHLC (20-period)
    donchian_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid_20 = (donchian_high_20 + donchian_low_20) / 2.0
    
    # Align 4h Donchian levels to 4h timeframe (no shift needed as we use prior completed 4h bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid_20)
    
    # Get 4h data for volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[high[0] - low[0]], tr])  # first TR is high-low
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Choppiness Index(14) on 4h data
    chop_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    denominator = highest_high - lowest_low
    denominator = np.where(denominator == 0, 1e-10, denominator)
    chop = 100 * np.log10(chop_sum / denominator) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_4h, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)  # Donchian needs 20, ATR/CHOP needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr14[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_atr = atr14[i]
        curr_chop = chop_aligned[i]
        
        # Volume confirmation: 2.0x threshold for strict entry
        vol_confirm = curr_volume > 2.0 * vol_ma[i]
        # Choppiness regime filter: > 61.8 = ranging market (favorable for mean reversion breakouts)
        chop_filter = curr_chop > 61.8
        
        if position == 0:
            # Check for entry signals
            if vol_confirm and chop_filter:
                # Long: Close breaks above Donchian upper
                if curr_close > donchian_high_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Short: Close breaks below Donchian lower
                elif curr_close < donchian_low_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        elif position == 1:
            # Long position: check exit conditions
            # Stoploss: 2.0 * ATR below entry
            stoploss = entry_price - 2.0 * curr_atr
            # Profit take: close below Donchian middle
            if curr_close < stoploss or curr_close < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: check exit conditions
            # Stoploss: 2.0 * ATR above entry
            stoploss = entry_price + 2.0 * curr_atr
            # Profit take: close above Donchian middle
            if curr_close > stoploss or curr_close > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0