#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and choppiness regime filter
# - Donchian(20): upper/lower channel from 20-period high/low
# - Long: price breaks above upper Donchian band + volume > 1.5x 20-period average + chop > 61.8 (range)
# - Short: price breaks below lower Donchian band + volume > 1.5x 20-period average + chop > 61.8 (range)
# - Uses 1d timeframe for choppiness calculation to avoid whipsaws
# - Discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits
# - Donchian breakouts capture strong momentum moves
# - Volume confirmation filters out weak breakouts
# - Choppiness regime filter (CHOP > 61.8) ensures we only trade in ranging markets where mean reversion works
# - Works in both bull (breakouts in ranging markets) and bear (breakouts in ranging markets) environments

name = "4h_1d_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for choppiness calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d choppiness index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range calculation
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # First period has no TR
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(atr14) / (hh14 - ll14)) / log10(14)
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    denominator = hh_14 - ll_14
    # Avoid division by zero
    denominator = np.where(denominator == 0, np.nan, denominator)
    chop = 100 * np.log10(sum_atr_14 / denominator) / np.log10(14)
    
    # Align choppiness to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Pre-compute 4h Donchian channels (20-period)
    donch_h_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_l_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donch_h_20[i]) or np.isnan(donch_l_20[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        volume_current = volume[i]
        
        # Donchian breakout conditions
        breakout_up = price_close > donch_h_20[i]
        breakout_down = price_close < donch_l_20[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Choppiness regime filter: CHOP > 61.8 indicates ranging market (mean reversion favorable)
        chop_regime = chop_aligned[i] > 61.8
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: breakout above upper Donchian band + volume confirmation + chop regime
        if breakout_up and vol_confirm and chop_regime:
            enter_long = True
        
        # Short: breakout below lower Donchian band + volume confirmation + chop regime
        if breakout_down and vol_confirm and chop_regime:
            enter_short = True
        
        # Exit conditions: price returns to middle of Donchian channel or breakout fails
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to or below the midpoint of the channel
            midpoint = (donch_h_20[i] + donch_l_20[i]) / 2
            exit_long = price_close <= midpoint
        elif position == -1:
            # Exit short if price returns to or above the midpoint of the channel
            midpoint = (donch_h_20[i] + donch_l_20[i]) / 2
            exit_short = price_close >= midpoint
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals