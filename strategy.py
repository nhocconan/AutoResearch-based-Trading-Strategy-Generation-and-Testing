#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ChoppinessIndex_DonchianBreakout_12hTrend"
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
    
    # Get 12h data once for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # 12h EMA34 trend filter
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_12h = (close_12h > ema34_12h).astype(float)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # Get 4h data for Choppiness Index (calculate on 4h data itself)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate Choppiness Index on 4h data (period=14)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    # ATR(14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum(ATR14) / (HH - LL)) / log10(14)
    # Add small epsilon to avoid division by zero
    denominator = highest_high - lowest_low
    denominator = np.where(denominator == 0, 1e-10, denominator)
    chop = 100 * np.log10(sum_atr14 / denominator) / np.log10(14)
    
    # Align 4h Chop to lower timeframe (already 4h, but need to align for consistency)
    chop_aligned = align_htf_to_ltf(prices, df_4h, chop)
    
    # Donchian channels on 4h (period=20)
    highest_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high = align_htf_to_ltf(prices, df_4h, highest_high_20)
    donchian_low = align_htf_to_ltf(prices, df_4h, lowest_low_20)
    
    # Volume spike detection: current volume > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(trend_12h_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above Donchian high + low chop (trending) + volume spike + 12h uptrend
            long_cond = (close[i] > donchian_high[i] and chop_aligned[i] < 38.2 and vol_spike[i] and trend_12h_aligned[i] > 0.5)
            
            # Short entry: price breaks below Donchian low + low chop (trending) + volume spike + 12h downtrend
            short_cond = (close[i] < donchian_low[i] and chop_aligned[i] < 38.2 and vol_spike[i] and trend_12h_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low (reversal) OR chop becomes high (range)
            if close[i] < donchian_low[i] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high OR chop becomes high (range)
            if close[i] > donchian_high[i] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Choppiness Index regime filter combined with Donchian breakouts on 4h timeframe.
# Uses 12h EMA34 for trend filter. Enters long when price breaks above Donchian(20) high
# with chop < 38.2 (trending market), volume spike, and 12h uptrend. Enters short when price
# breaks below Donchian low with chop < 38.2, volume spike, and 12h downtrend.
# Exits when price reverses through opposite Donchian level OR chop > 61.8 (range market).
# Chop < 38.2 = trending (trend follow), Chop > 61.8 = range (mean revert would be better but we follow trend here).
# Volume confirmation (2.0x 20-period average) reduces false breakouts.
# Targets 20-35 trades/year on 4h timeframe to avoid overtrading. Works in both bull and bear markets
# by only taking trades in trending regimes (chop < 38.2) and using multi-timeframe trend alignment.
# Uses discrete sizing (0.25) to minimize churn from frequent signal changes.