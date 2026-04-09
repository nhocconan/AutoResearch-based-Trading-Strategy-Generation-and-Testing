#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d ATR-based volatility regime filter and 1w Donchian breakout for direction
# - Uses 1d HTF for ATR-based volatility regime: high volatility (ATR>ATR_ma) enables breakout trading
# - Uses 1w HTF for Donchian(20) breakout: price > 20-period high = bullish, < 20-period low = bearish
# - In high volatility regime: trade breakouts in direction of weekly Donchian trend
# - In low volatility regime: fade weekly extremes (mean reversion at Donchian bands)
# - Volume confirmation: current 6h volume > 1.5x 20-period average to avoid low-volume false signals
# - Fixed position size 0.25 to control drawdown
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1d_1w_atr_regime_donchian_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d ATR(14) and its 20-period moving average for volatility regime
    tr1 = high_1d[1:] - low_1d[:-1]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w Donchian(20) channels
    donchian_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high_20 + donchian_low_20) / 2
    
    # Align all HTF data to 6h timeframe (wait for completed HTF bar)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_20_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20)
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma_20_aligned[i]) or
            np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0 or atr_ma_20_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Volatility regime: high volatility when ATR > ATR_MA
        high_volatility = atr_1d_aligned[i] > atr_ma_20_aligned[i]
        
        # Weekly Donchian levels
        weekly_high = donchian_high_20_aligned[i]
        weekly_low = donchian_low_20_aligned[i]
        weekly_mid = donchian_mid_aligned[i]
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit conditions
            if high_volatility:
                # In high vol regime: exit when price breaks below weekly low or trend changes
                if close[i] < weekly_low or close[i] < weekly_mid:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            else:
                # In low vol regime: exit when price reverts to weekly mid
                if close[i] > weekly_mid:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
                    
        elif position == -1:  # Short position
            # Exit conditions
            if high_volatility:
                # In high vol regime: exit when price breaks above weekly high or trend changes
                if close[i] > weekly_high or close[i] > weekly_mid:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
            else:
                # In low vol regime: exit when price reverts to weekly mid
                if close[i] < weekly_mid:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
        else:  # Flat
            # Entry logic based on volatility regime and price action
            if volume_confirmed:
                if high_volatility:
                    # High volatility regime: trade breakouts
                    if close[i] > weekly_high:
                        # Break above weekly high: long
                        position = 1
                        signals[i] = position_size
                    elif close[i] < weekly_low:
                        # Break below weekly low: short
                        position = -1
                        signals[i] = -position_size
                else:
                    # Low volatility regime: mean reversion at extremes
                    if close[i] < weekly_low and close[i] < weekly_mid:
                        # Near weekly low: long mean reversion
                        position = 1
                        signals[i] = position_size
                    elif close[i] > weekly_high and close[i] > weekly_mid:
                        # Near weekly high: short mean reversion
                        position = -1
                        signals[i] = -position_size
    
    return signals