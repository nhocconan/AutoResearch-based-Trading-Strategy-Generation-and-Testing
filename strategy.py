#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and chop regime filter
# - Primary: 12h timeframe for lower trade frequency (~12-37/year target)
# - HTF: 1d for volume spike confirmation and choppiness regime
# - Long: Price breaks above Donchian(20) high + 1d volume > 2x 20-day MA + CHOP > 61.8 (range)
# - Short: Price breaks below Donchian(20) low + 1d volume > 2x 20-day MA + CHOP > 61.8 (range)
# - Exit: Price reverts to Donchian(20) midpoint or opposite breakout
# - Position sizing: 0.25 (discrete level to balance return and drawdown)
# - Target: 50-150 total trades over 4 years (12-37/year) - within 12h sweet spot
# - Donchian breakouts work in both bull/bear markets when combined with regime filter
# - Volume confirmation increases breakout reliability
# - CHOP > 61.8 ensures we only trade in ranging markets (avoid strong trends where breakouts fail)

name = "12h_1d_donchian_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 12h OHLCV
    open_12h = prices['open'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    volume_12h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 12h Donchian(20) channels
    # Highest high of last 20 periods
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lowest low of last 20 periods
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    # Midpoint for exit
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 1d True Range for ATR
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR for choppiness indicator components
    atr_sum_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    # Calculate 1d true range sum for denominator
    tr_sum_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    # Calculate 1d max high - min low over 14 periods
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Choppiness Index: CHOP = 100 * log10(atr_sum_14 / tr_sum_14) / log10(14)
    # Avoid division by zero
    chop_ratio = np.where(tr_sum_14 > 0, atr_sum_14 / tr_sum_14, 1.0)
    chop = 100 * np.log10(chop_ratio) / np.log10(14)
    chop = np.where(range_14 > 0, chop, 50.0)  # Default to middle when no range
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Session filter: 00-23 UTC (trade all hours for 12h timeframe)
    # For 12h, we can trade all hours as each bar represents half a day
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # Volume confirmation: current 1d volume > 2x 20-day MA (aligned to 12h)
        volume_spike = volume_1d[i // 1] > 2.0 * volume_ma_20_1d_aligned[i] if i < len(volume_1d) else False
        
        # Chop regime: CHOP > 61.8 indicates ranging market (good for mean reversion after breakout)
        chop_regime = chop_aligned[i] > 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian high + volume spike + chop regime
            if close_12h[i] > donchian_high[i] and volume_spike and chop_regime:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low + volume spike + chop regime
            elif close_12h[i] < donchian_low[i] and volume_spike and chop_regime:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price reverts to Donchian midpoint (mean reversion)
            # 2. Price breaks opposite Donchian level (stop and reverse)
            
            if position == 1:  # Long position
                exit_condition = (
                    close_12h[i] < donchian_mid[i] or  # Reverted to midpoint
                    close_12h[i] < donchian_low[i]     # Break below low (stop and reverse)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    close_12h[i] > donchian_mid[i] or  # Reverted to midpoint
                    close_12h[i] > donchian_high[i]    # Break above high (stop and reverse)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals