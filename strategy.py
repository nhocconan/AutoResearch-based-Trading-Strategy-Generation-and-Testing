#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d volume confirmation and 1w chop regime filter
# - Long: price breaks above Donchian(20) high, 1d volume > 1.5x 20-period average, 1w chop < 61.8 (trending regime)
# - Short: price breaks below Donchian(20) low, 1d volume > 1.5x 20-period average, 1w chop < 61.8 (trending regime)
# - Exit: price returns to Donchian midpoint or ATR-based stop (2.5 ATR)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Donchian breakouts capture momentum in trending markets
# - Volume confirmation filters false breakouts
# - Chop regime filter ensures we only trade in trending conditions (avoids whipsaw in ranging markets)

name = "12h_1d_1w_donchian_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load 1w data ONCE before loop for chop regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Pre-compute 1w Chop Index(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr_1w = np.maximum(high_1w - low_1w, np.maximum(np.abs(high_1w - np.roll(close_1w, 1)), np.abs(low_1w - np.roll(close_1w, 1))))
    tr_1w[0] = high_1w[0] - low_1w[0]
    
    # Sum of ATR over 14 periods
    sum_tr_14 = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    highest_high_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Chop Index = 100 * log10(sum_tr_14 / (highest_high_14 - lowest_low_14)) / log10(14)
    range_14 = highest_high_14 - lowest_low_14
    chop = np.where(range_14 > 0, 100 * np.log10(sum_tr_14 / range_14) / np.log10(14), 50)
    chop = np.where(np.isnan(chop) | np.isinf(chop), 50, chop)
    
    # Align 1w Chop to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Pre-compute 1d volume SMA(20)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute Donchian channels for 12h timeframe
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high_20 + lowest_low_20) / 2
    
    # Pre-compute ATR for stoploss (12h timeframe)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(chop_aligned[i]) or np.isnan(volume_sma_20_aligned[i]) or
            np.isnan(donchian_mid[i]) or np.isnan(atr_20[i]) or
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        high_price = high[i]
        low_price = low[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Regime filter: 1w Chop < 61.8 (trending regime, not ranging)
        chop_trend = chop_aligned[i] < 61.8
        
        # Donchian breakout levels
        upper_band = highest_high_20[i]
        lower_band = lowest_low_20[i]
        mid_band = donchian_mid[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price breaks above Donchian upper band
        if close_price > upper_band and vol_confirm and chop_trend:
            enter_long = True
        
        # Short breakout: price breaks below Donchian lower band
        if close_price < lower_band and vol_confirm and chop_trend:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to midpoint or ATR-based stop
            exit_long = (close_price <= mid_band) or (close_price <= entry_price - 2.5 * atr_20[i])
        elif position == -1:
            # Exit short if price returns to midpoint or ATR-based stop
            exit_short = (close_price >= mid_band) or (close_price >= entry_price + 2.5 * atr_20[i])
        
        # Track entry price for stoploss calculation
        if enter_long or enter_short:
            entry_price = close_price
        
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