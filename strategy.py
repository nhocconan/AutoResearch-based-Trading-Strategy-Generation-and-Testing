#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_RegimeFilter_VolumeSpike
Hypothesis: 4h Donchian(20) breakout aligned with 1d EMA50 trend, filtered by choppiness regime (CHOP < 38.2 = trending) and volume spike (>1.5x avg). Long on upper band break, short on lower band break. Exit on opposite band break or loss of trend/regime. Designed for 20-40 trades/year on 4h to minimize fee drag while capturing strong trending moves. Works in bull markets (breakouts with 1d uptrend) and bear markets (breakdowns with 1d downtrend). Uses discrete position sizing (0.25) to reduce churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 4h Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Choppiness regime filter (CHOP < 38.2 = trending)
    atr_series = pd.Series(high - low).rolling(window=14, min_periods=14).mean()
    max_high = high_series.rolling(window=14, min_periods=14).max()
    min_low = low_series.rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_series.sum() / (max_high - min_low)) / np.log10(14)
    chop_values = chop.values
    chop_aligned = chop_values  # already LTF
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # 25% position size
    
    # Warmup: need enough for 1d EMA50 (~50 1d bars = ~200 4h bars), Donchian(20), CHOP(14), volume avg
    start_idx = max(200, 20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        ema_val = ema_50_aligned[i]
        chop_val = chop_aligned[i]
        vol_spike = volume_spike[i]
        
        # Regime filter: only trade in trending markets (CHOP < 38.2)
        in_trending_regime = chop_val < 38.2
        
        if position == 0:
            # Flat - look for entry: Donchian breakout with 1d EMA50 alignment, volume spike, and trending regime
            # Long: Close > Donchian upper AND price > 1d EMA50 AND volume spike AND trending regime
            # Short: Close < Donchian lower AND price < 1d EMA50 AND volume spike AND trending regime
            long_condition = (close_val > donchian_high_val and 
                            close_val > ema_val and 
                            vol_spike and 
                            in_trending_regime)
            short_condition = (close_val < donchian_low_val and 
                             close_val < ema_val and 
                             vol_spike and 
                             in_trending_regime)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when price breaks below Donchian lower OR loses 1d EMA50 alignment OR exits trending regime
            if (close_val < donchian_low_val or 
                close_val < ema_val or 
                not in_trending_regime):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above Donchian upper OR loses 1d EMA50 alignment OR exits trending regime
            if (close_val > donchian_high_val or 
                close_val > ema_val or 
                not in_trending_regime):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_RegimeFilter_VolumeSpike"
timeframe = "4h"
leverage = 1.0