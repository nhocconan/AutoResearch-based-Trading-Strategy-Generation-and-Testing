#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeSpike_ChopRegime
Hypothesis: Uses 4h Donchian(20) breakout for entries, filtered by 1d EMA50 trend, volume spike (>1.5x average), and choppiness regime (CHOP > 61.8 = range -> mean revert, CHOP < 38.2 = trending -> trend follow). In trending regime, breakout in direction of 1d trend; in ranging regime, fade Donchian breaks. Uses 4h timeframe with tight entries to avoid fee drag: target 20-50 trades/year. Donchian breakouts capture momentum; 1d EMA50 filters higher-timeframe trend; volume confirmation avoids false breakouts; chop regime adapts to market conditions. Works in bull markets via trend-following breaks and in bear markets via mean-reversion fades in ranging regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and chop regime
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d choppiness index (CHOP)
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high, n) - min(low, n))) / log10(n)
    # Using ATR(14) and window=14 for chop calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[tr1[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Chop calculation: sum of ATR over period / (max high - min low) over same period
    chop_window = 14
    sum_atr_1d = pd.Series(atr_1d).rolling(window=chop_window, min_periods=chop_window).sum().values
    max_high_1d = pd.Series(high_1d).rolling(window=chop_window, min_periods=chop_window).max().values
    min_low_1d = pd.Series(low_1d).rolling(window=chop_window, min_periods=chop_window).min().values
    range_1d = max_high_1d - min_low_1d
    
    # Avoid division by zero
    chop_1d = np.where(range_1d > 0, 100 * np.log10(sum_atr_1d / range_1d) / np.log10(chop_window), 50)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 4h Donchian(20) channels
    donchian_window = 20
    highest_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need Donchian(20), 1d EMA50 (50), 1d ATR(14) for chop (14+14=28), volume avg (20)
    start_idx = max(donchian_window, 50, 28, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        donchian_high = highest_high[i]
        donchian_low = lowest_low[i]
        ema_1d_val = ema_50_1d_aligned[i]
        chop_val = chop_1d_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Determine regime: chop > 61.8 = ranging (mean revert), chop < 38.2 = trending (trend follow)
            is_ranging = chop_val > 61.8
            is_trending = chop_val < 38.2
            
            if is_trending:
                # Trending regime: breakout in direction of 1d trend
                # Long: price breaks above Donchian high AND 1d uptrend AND volume confirmation
                long_condition = (high_val > donchian_high) and (close_val > ema_1d_val) and vol_conf
                # Short: price breaks below Donchian low AND 1d downtrend AND volume confirmation
                short_condition = (low_val < donchian_low) and (close_val < ema_1d_val) and vol_conf
                
                if long_condition:
                    signals[i] = size
                    position = 1
                elif short_condition:
                    signals[i] = -size
                    position = -1
            elif is_ranging:
                # Ranging regime: fade Donchian breaks (mean reversion)
                # Long: price breaks below Donchian low AND 1d uptrend (buy dips in uptrend) AND volume confirmation
                long_condition = (low_val < donchian_low) and (close_val > ema_1d_val) and vol_conf
                # Short: price breaks above Donchian high AND 1d downtrend (sell rallies in downtrend) AND volume confirmation
                short_condition = (high_val > donchian_high) and (close_val < ema_1d_val) and vol_conf
                
                if long_condition:
                    signals[i] = size
                    position = 1
                elif short_condition:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: Donchian breakout failed or trend change
            # Exit if price returns to Donchian mid-point or 1d trend breaks
            donchian_mid = (donchian_high + donchian_low) / 2
            exit_condition = (close_val < donchian_mid) or (close_val < ema_1d_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Donchian breakout failed or trend change
            # Exit if price returns to Donchian mid-point or 1d trend breaks
            donchian_mid = (donchian_high + donchian_low) / 2
            exit_condition = (close_val > donchian_mid) or (close_val > ema_1d_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0