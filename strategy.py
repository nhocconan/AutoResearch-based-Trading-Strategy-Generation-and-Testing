#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA200 trend filter + volume confirmation + chop regime filter
# Donchian breakouts capture strong momentum moves; EMA200 ensures we trade with higher timeframe trend
# Volume confirmation (1.5x 20-period avg) filters weak breakouts
# Chop regime filter (CHOP > 61.8 = range, CHOP < 38.2 = trending) avoids whipsaws in sideways markets
# Works in bull/bear: EMA200 trend filter avoids counter-trend whipsaws in ranging markets
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.25-0.30

name = "4h_1d_donchian_ema200_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA200 trend filter and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend direction
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 1d chop regime (using 14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_series = pd.Series(close_1d)
    atr_14 = pd.Series(
        np.maximum(
            np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1])),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    ).rolling(window=14, min_periods=14).mean().values
    atr_14 = np.concatenate([np.full(14, np.nan), atr_14])
    
    true_range_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    price_change = np.abs(close_1d[14:] - close_1d[:-14])
    price_change = np.concatenate([np.full(14, np.nan), price_change])
    
    chop = np.full(len(close_1d), np.nan)
    valid = ~(np.isnan(true_range_sum) | np.isnan(price_change) | (true_range_sum == 0))
    chop[valid] = 100 * np.log10(price_change[valid] / true_range_sum[valid]) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(avg_volume[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        # Chop regime filter: only trade when trending (CHOP < 38.2)
        trending_regime = chop_aligned[i] < 38.2
        
        if position == 1:  # Long position
            # Exit: price < Donchian low OR price < 1d EMA200 (trend change)
            if close[i] < donchian_low[i] or close[i] < ema_200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Donchian high OR price > 1d EMA200 (trend change)
            if close[i] > donchian_high[i] or close[i] > ema_200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation, Donchian breakout, EMA200 trend filter, and chop regime
            if volume_confirmed and trending_regime:
                # Long entry: price > Donchian high AND price > 1d EMA200 (bullish breakout + uptrend)
                if close[i] > donchian_high[i] and close[i] > ema_200_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < Donchian low AND price < 1d EMA200 (bearish breakout + downtrend)
                elif close[i] < donchian_low[i] and close[i] < ema_200_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals