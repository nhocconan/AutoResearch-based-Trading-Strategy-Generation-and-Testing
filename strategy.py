#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and chop regime filter
# - Primary: 4h timeframe (optimal trade frequency, proven winners)
# - HTF: 1d for volume confirmation (20-period MA) and chop regime (14-period)
# - Long: Price breaks above Donchian(20) high + 1d volume > 1.5x 20-period MA + chop < 61.8 (trending)
# - Short: Price breaks below Donchian(20) low + 1d volume > 1.5x 20-period MA + chop < 61.8 (trending)
# - Exit: Donchian(10) opposite break (trailing stop) or ATR(14) stoploss (2.0)
# - Position sizing: 0.25 (discrete level)
# - Works in bull/bear: Donchian breakouts catch trends, chop filter avoids whipsaws in ranging markets (2025)
# - Target: 80-180 total trades over 4 years (20-45/year) - within 4h sweet spot

name = "4h_1d_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
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
    
    # Calculate 4h Donchian channels (20-period for entry, 10-period for exit)
    high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    high_10 = pd.Series(high_4h).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low_4h).rolling(window=10, min_periods=10).min().values
    
    # Calculate 1d ATR(14) for stoploss
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(TR14) / (ATR14 * 14)) / log10(14)
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).mean().values
    chop_raw = 100 * np.log10(sum_tr_14 / (atr_14 * 14)) / np.log10(14)
    # Handle division by zero or invalid values
    chop = np.where((atr_14 * 14) > 0, chop_raw, 50.0)  # Default to 50 (neutral) if invalid
    
    # Align 1d indicators to 4h
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0  # Track entry price for ATR-based stoploss
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(high_10[i]) or np.isnan(low_10[i]) or
            np.isnan(volume_ma_20_1d_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian(20) high + volume spike + trending regime (chop < 61.8)
            if (close_4h[i] > high_20[i] and 
                volume_4h[i] > 1.5 * volume_ma_20_1d_aligned[i] and 
                chop_aligned[i] < 61.8):
                position = 1
                entry_price = close_4h[i]
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian(20) low + volume spike + trending regime (chop < 61.8)
            elif (close_4h[i] < low_20[i] and 
                  volume_4h[i] > 1.5 * volume_ma_20_1d_aligned[i] and 
                  chop_aligned[i] < 61.8):
                position = -1
                entry_price = close_4h[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Donchian(10) opposite break (trailing stop)
            # 2. ATR-based stoploss (2.0 * ATR from entry)
            
            if position == 1:  # Long position
                donchian_exit = close_4h[i] < low_10[i]  # Break below Donchian(10) low
                atr_stop = close_4h[i] < entry_price - 2.0 * atr_1d_aligned[i]  # 2*ATR stoploss
                exit_condition = donchian_exit or atr_stop
                if exit_condition:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                donchian_exit = close_4h[i] > high_10[i]  # Break above Donchian(10) high
                atr_stop = close_4h[i] > entry_price + 2.0 * atr_1d_aligned[i]  # 2*ATR stoploss
                exit_condition = donchian_exit or atr_stop
                if exit_condition:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals