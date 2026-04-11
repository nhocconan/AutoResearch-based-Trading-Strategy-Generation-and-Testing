#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return signals
    
    # Calculate weekly data from 1d bars (weekly pivot points)
    # We'll group by week using the index
    weekly_high = df_1d['high'].resample('W').max()
    weekly_low = df_1d['low'].resample('W').min()
    weekly_close = df_1d['close'].resample('W').last()
    
    # Calculate weekly pivot points
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    
    # Align weekly data to 1d index
    weekly_pivot_1d = weekly_pivot.reindex(df_1d.index, method='ffill').values
    weekly_r1_1d = weekly_r1.reindex(df_1d.index, method='ffill').values
    weekly_s1_1d = weekly_s1.reindex(df_1d.index, method='ffill').values
    weekly_r2_1d = weekly_r2.reindex(df_1d.index, method='ffill').values
    weekly_s2_1d = weekly_s2.reindex(df_1d.index, method='ffill').values
    
    # Align weekly pivots to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot_1d)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1_1d)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1_1d)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1d, weekly_r2_1d)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1d, weekly_s2_1d)
    
    # Volume confirmation: 20-period average on 6h
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volatility filter: 6h ATR ratio
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_6h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_20 = pd.Series(atr_6h).rolling(window=20, min_periods=20).mean().values
    atr_ratio = np.where(atr_ma_20 > 0, atr_6h / atr_ma_20, 1.0)
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume_current > 1.3 * volume_sma_20[i]
        
        # Volatility regime: trade only when volatility is elevated
        vol_regime = atr_ratio[i] > 0.7
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price breaks above weekly R1 with volume and volatility
        price_above_r1 = price_close > weekly_r1_aligned[i]
        if price_above_r1 and vol_confirm and vol_regime:
            enter_long = True
        
        # Short: Price breaks below weekly S1 with volume and volatility
        price_below_s1 = price_close < weekly_s1_aligned[i]
        if price_below_s1 and vol_confirm and vol_regime:
            enter_short = True
        
        # Exit conditions: price crosses back through weekly pivot
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price crosses below weekly pivot
            exit_long = price_close < weekly_pivot_aligned[i]
        elif position == -1:
            # Exit short if price crosses above weekly pivot
            exit_short = price_close > weekly_pivot_aligned[i]
        
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