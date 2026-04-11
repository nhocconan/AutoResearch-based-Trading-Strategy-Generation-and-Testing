#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + volume confirmation + choppiness regime filter
# - Donchian(20) breakout: price closes above/below 20-period high/low
# - Volume confirmation: current volume > 1.5x 20-period average volume
# - Choppiness regime: CHOP(14) > 61.8 = ranging (mean reversion), CHOP < 38.2 = trending (breakout)
# - Long: price > Donchian high AND volume confirmation AND CHOP < 38.2
# - Short: price < Donchian low AND volume confirmation AND CHOP < 38.2
# - Exit: reverse Donchian breakout or CHOP > 61.8 (choppy market)
# - Position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 75-200 total trades over 4 years (19-50/year) to stay within fee drag limits
# - Works in bull markets (breakouts continue) and bear markets (breakdowns continue)

name = "4h_donchian_volume_chop_v1"
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
    
    # Load 1d data ONCE before loop for choppiness regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d choppiness index
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original indices
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR14)/ (HH14-LL14)) / log10(14)
    sum_atr = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr / (hh_14 - ll_14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute Donchian channels on 4h timeframe
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    for i in range(donchian_period, n):  # Start after Donchian warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_current = close[i]
        volume_current = volume[i]
        
        # Donchian breakout conditions
        breakout_high = close_current > donchian_high[i]
        breakout_low = close_current < donchian_low[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Regime filter: only trade in trending markets (CHOP < 38.2)
        trending_regime = chop_aligned[i] < 38.2
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below Donchian low OR market becomes choppy
            exit_long = (close_current < donchian_low[i]) or (chop_aligned[i] > 61.8)
        elif position == -1:
            # Exit short if price breaks above Donchian high OR market becomes choppy
            exit_short = (close_current > donchian_high[i]) or (chop_aligned[i] > 61.8)
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Donchian breakout high + volume confirmation + trending regime
        if breakout_high and vol_confirm and trending_regime:
            enter_long = True
        
        # Short: Donchian breakout low + volume confirmation + trending regime
        if breakout_low and vol_confirm and trending_regime:
            enter_short = True
        
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