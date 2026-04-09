#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Donchian breakout for trend and 1d ATR expansion for momentum confirmation
# - Uses 12h HTF for Donchian channels: breakout above/below 20-period high/low determines trend direction
# - Uses 1d HTF for ATR ratio: current ATR(14) > 1.5x ATR(50) indicates expanding volatility/momentum
# - In bullish trend (price > 12h Donchian upper): look for long entries when ATR expansion occurs
# - In bearish trend (price < 12h Donchian lower): look for short entries when ATR expansion occurs
# - Volume filter: current 6h volume > 1.2x 20-period average to avoid low-quality breakouts
# - Fixed position size 0.25 to control drawdown
# - Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years)

name = "6h_12h_1d_donchian_atr_volume_v1"
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
    
    # Load 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h Donchian channels (20 periods)
    period20_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_upper = period20_high
    donchian_lower = period20_low
    
    # Calculate 1d ATR for volatility expansion filter
    # True Range = max(high-low, abs(high-close_prev), abs(low-close_prev))
    tr1 = pd.Series(high_1d - low_1d).abs()
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1))).abs()
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1))).abs()
    tr1.iloc[0] = high_1d[0] - low_1d[0]  # First period TR
    tr2.iloc[0] = np.abs(high_1d[0] - close_1d[0])  # Assume close_prev = close for first bar
    tr3.iloc[0] = np.abs(low_1d[0] - close_1d[0])
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    atr_14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(true_range).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / (atr_50 + 1e-10)  # Current ATR / longer ATR
    
    # Align all HTF data to 6h timeframe (wait for completed HTF bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.2x average
        volume_confirmed = volume[i] > 1.2 * vol_ma_20[i]
        
        # ATR expansion: current volatility > 1.5x longer-term average
        atr_expansion = atr_ratio_aligned[i] > 1.5
        
        # Trend determination: price relative to 12h Donchian channels
        bullish_breakout = close[i] > donchian_upper_aligned[i]
        bearish_breakout = close[i] < donchian_lower_aligned[i]
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit conditions: trend reversal or loss of momentum
            if not bullish_breakout or not atr_expansion:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit conditions: trend reversal or loss of momentum
            if not bearish_breakout or not atr_expansion:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Entry logic: breakout with volume and momentum confirmation
            if volume_confirmed and atr_expansion:
                if bullish_breakout:
                    position = 1
                    signals[i] = position_size
                elif bearish_breakout:
                    position = -1
                    signals[i] = -position_size
    
    return signals