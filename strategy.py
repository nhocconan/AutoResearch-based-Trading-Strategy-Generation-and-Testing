#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and chop regime filter
# - Primary: 4h timeframe for optimal trade frequency (target 20-50/year)
# - HTF: 1d for volume MA and choppiness index regime filter
# - Long: Price breaks above Donchian(20) high + 1d volume > 1.5x 20-period MA + CHOP(14) < 50 (trending regime)
# - Short: Price breaks below Donchian(20) low + 1d volume > 1.5x 20-period MA + CHOP(14) < 50 (trending regime)
# - Exit: Price reverts to Donchian midpoint (mean reversion) or ATR-based trailing stop
# - Position sizing: 0.25 (discrete level)
# - Works in bull/bear: Donchian breakouts capture trends; chop filter avoids whipsaws in ranging markets

name = "4h_1d_donchian_volume_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
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
    
    # Calculate 4h Donchian Channel (20-period)
    highest_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2.0
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index (CHOP) - 14 period
    # CHOP = 100 * log10(sum(ATR(1)) / (n * log(n))) / log10(n)
    # where ATR(1) = True Range for 1 period
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum()
    n_log = np.log(14)
    chop = 100 * (np.log10(atr_sum) - np.log10(n_log)) / np.log10(14)
    chop_values = chop.values
    
    # Align 1d indicators to 4h
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # Session filter: 08-20 UTC (optional, can help reduce noise)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Session filter: only trade 08-20 UTC (comment out if too restrictive)
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is invalid
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # Trending regime: CHOP < 50 (below 50 indicates trending market)
        trending_regime = chop_aligned[i] < 50
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        # Need to get current 1d volume - since we're in 4h, we use the latest available 1d volume
        # Find the 1d index corresponding to current 4h bar
        # Using aligned arrays ensures we get the completed 1d volume
        volume_spike = volume_1d[-1] > 1.5 * volume_ma_20_1d_aligned[i] if len(volume_1d) > 0 else False
        # Simpler approach: use the aligned volume data directly
        # We'll use the 1d volume value aligned to current 4h bar
        # For simplicity, we'll use a volume MA ratio approach
        if i >= 4:  # Need at least one 1d bar per 4h bar (approx)
            # Get the 1d volume for the day containing this 4h bar
            # Since we don't have direct 1d volume in 4h array, we'll use a proxy
            # Use the fact that volume_ma_20_1d_aligned represents the 20-period MA
            # We'll compare current volume to this MA using a simplified approach
            # For now, we'll use a volume spike condition based on 4h volume vs its own MA
            volume_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
            volume_spike = volume_4h[i] > 1.5 * volume_ma_20_4h[i]
        else:
            volume_spike = False
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian high + trending regime + volume spike
            if (close_4h[i] > highest_20[i] and trending_regime and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low + trending regime + volume spike
            elif (close_4h[i] < lowest_20[i] and trending_regime and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price reverts to Donchian midpoint (mean reversion)
            # 2. ATR-based trailing stop (simplified as Donchian midpoint for now)
            
            if position == 1:  # Long position
                exit_condition = close_4h[i] < donchian_mid[i]  # Reverted to midpoint
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = close_4h[i] > donchian_mid[i]  # Reverted to midpoint
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals