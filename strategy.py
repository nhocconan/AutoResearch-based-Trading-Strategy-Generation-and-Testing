#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_v3"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return signals
    
    # Calculate 1d ATR for volatility filter (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 1d ATR MA (20-period) for volatility regime filter
    atr_ma_20_1d = pd.Series(atr_14_1d_aligned).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ATR ratio (current / MA) for regime detection
    atr_ratio_1d = np.where(atr_ma_20_1d > 0, atr_14_1d_aligned / atr_ma_20_1d, 1.0)
    
    # Calculate Camarilla levels on 1d data (using previous 1d bar's range)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_H4 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_L4 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    
    # Volume confirmation: 20-period average on 4h
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Price momentum filter: close above/below 20-period SMA on 4h
    close_sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_H4_aligned[i]) or np.isnan(camarilla_L4_aligned[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(atr_ratio_1d[i]) or
            np.isnan(close_sma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Volatility regime filter: trade only when volatility is elevated (ATR ratio > 0.8)
        vol_regime = atr_ratio_1d[i] > 0.8
        
        # Price momentum filter: price above SMA for longs, below SMA for shorts
        mom_filter_long = price_close > close_sma_20[i]
        mom_filter_short = price_close < close_sma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price breaks above Camarilla H4 level + volume confirmation + volatility regime + momentum filter
        if price_close > camarilla_H4_aligned[i] and vol_confirm and vol_regime and mom_filter_long:
            enter_long = True
        
        # Short: Price breaks below Camarilla L4 level + volume confirmation + volatility regime + momentum filter
        if price_close < camarilla_L4_aligned[i] and vol_confirm and vol_regime and mom_filter_short:
            enter_short = True
        
        # Exit conditions: price crosses back through the Camarilla mid-point (C level)
        exit_long = False
        exit_short = False
        
        # Calculate Camarilla C level (close of previous 1d bar)
        camarilla_C = prev_close_1d
        camarilla_C_aligned = align_htf_to_ltf(prices, df_1d, camarilla_C)
        
        if position == 1:
            # Exit long if price crosses below Camarilla C level
            exit_long = price_close < camarilla_C_aligned[i]
        elif position == -1:
            # Exit short if price crosses above Camarilla C level
            exit_short = price_close > camarilla_C_aligned[i]
        
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