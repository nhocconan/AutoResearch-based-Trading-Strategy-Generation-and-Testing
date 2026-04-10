#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and chop regime filter
# - Primary: 4h timeframe for optimal trade frequency (target 20-50 trades/year)
# - Entry: Price breaks above/below Donchian(20) channel + 1d volume > 2x 20-period MA + chop < 61.8 (trending regime)
# - Exit: Opposite Donchian(10) break or ATR-based trailing stop (3x ATR)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - ATR stoploss: signal → 0 when price < highest high - 3*ATR (long) or price > lowest low + 3*ATR (short)
# - Designed to work in both bull (breakouts) and bear (breakdowns) markets
# - Volume confirmation avoids false breakouts
# - Chop filter ensures we only trade in trending markets
# - Target: 80-150 total trades over 4 years (20-37/year) - within 4h sweet spot

name = "4h_1d_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
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
    
    # Calculate 4h Donchian channels
    # Donchian(20) for entry: upper_20 = max(high, lookback=20), lower_20 = min(low, lookback=20)
    # Donchian(10) for exit: upper_10 = max(high, lookback=10), lower_10 = min(low, lookback=10)
    high_series = pd.Series(high_4h)
    low_series = pd.Series(low_4h)
    upper_20 = high_series.rolling(window=20, min_periods=20).max().values
    lower_20 = low_series.rolling(window=20, min_periods=20).min().values
    upper_10 = high_series.rolling(window=10, min_periods=10).max().values
    lower_10 = low_series.rolling(window=10, min_periods=10).min().values
    
    # Calculate 4h ATR(14) for stoploss
    tr1 = pd.Series(high_4h).shift(1) - pd.Series(low_4h).shift(1)
    tr2 = abs(pd.Series(high_4h) - pd.Series(close_4h).shift(1))
    tr3 = abs(pd.Series(low_4h) - pd.Series(close_4h).shift(1))
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h = tr_4h.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 1d Chopiness Index(14) for regime filter
    # Chop = 100 * log10(sum(ATR,14) / (max(high,14) - min(low,14))) / log10(14)
    # Chop > 61.8 = ranging, Chop < 38.2 = trending
    atr_1d_series = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    atr_1d_series = atr_1d_series.combine_first(abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1)))
    atr_1d_series = atr_1d_series.combine_first(abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1)))
    atr_1d = atr_1d_series.rolling(window=14, min_periods=14).mean().values
    
    max_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denominator = max_high_1d - min_low_1d
    # Avoid division by zero
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)
    chop_value = 100 * np.log10(pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values / chop_denominator) / np.log10(14)
    chop_value = np.where(chop_denominator == 0, 50, chop_value)  # Neutral when range=0
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_value)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_high = 0.0
    lowest_low = 0.0
    
    for i in range(60, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or 
            np.isnan(upper_10[i]) or np.isnan(lower_10[i]) or 
            np.isnan(atr_4h[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Update trailing stop levels
        if position == 1:  # Long position
            highest_high = max(highest_high, high_4h[i])
        elif position == -1:  # Short position
            lowest_low = min(lowest_low, low_4h[i]) if lowest_low != 0 else low_4h[i]
        
        # Regime conditions
        # Chop < 61.8 = trending regime (we want to trade in trending markets)
        trending_regime = chop_aligned[i] < 61.8
        
        # Volume confirmation: current 1d volume > 2x 20-period MA
        volume_spike = volume_1d[i // 16] > 2.0 * volume_ma_20_1d_aligned[i] if i >= 16 else False
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian(20) upper + volume spike + trending regime
            if close_4h[i] > upper_20[i] and volume_spike and trending_regime:
                position = 1
                entry_price = close_4h[i]
                highest_high = high_4h[i]
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian(20) lower + volume spike + trending regime
            elif close_4h[i] < lower_20[i] and volume_spike and trending_regime:
                position = -1
                entry_price = close_4h[i]
                lowest_low = low_4h[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            exit_condition = False
            
            if position == 1:  # Long position
                # Exit conditions:
                # 1. Price breaks below Donchian(10) lower (opposite breakout)
                # 2. ATR trailing stop: price < highest_high - 3*ATR
                if close_4h[i] < lower_10[i]:
                    exit_condition = True
                elif high_4h[i] < highest_high - 3.0 * atr_4h[i]:
                    exit_condition = True
                    
                if exit_condition:
                    position = 0
                    entry_price = 0.0
                    highest_high = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                # Exit conditions:
                # 1. Price breaks above Donchian(10) upper (opposite breakout)
                # 2. ATR trailing stop: price > lowest_low + 3*ATR
                if close_4h[i] > upper_10[i]:
                    exit_condition = True
                elif low_4h[i] > lowest_low + 3.0 * atr_4h[i]:
                    exit_condition = True
                    
                if exit_condition:
                    position = 0
                    entry_price = 0.0
                    lowest_low = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals