#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_1d_camarilla_range_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-calculate hour filter once
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return signals
    
    # Calculate 1d ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d ATR MA for volatility regime
    atr_ma_20_1d = pd.Series(atr_14_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio_1d = np.where(atr_ma_20_1d > 0, atr_14_1d / atr_ma_20_1d, 1.0)
    
    # Calculate Camarilla levels on 1d data (using previous day's range)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_H3 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_L3 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_H4 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d)  # More extreme level
    camarilla_L4 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d)
    
    # Align Camarilla levels to 1h timeframe
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    
    # Volume confirmation: 24-period average on 1h (1 day)
    volume_sma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    for i in range(100, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is invalid
        if (np.isnan(camarilla_H3_aligned[i]) or np.isnan(camarilla_L3_aligned[i]) or
            np.isnan(volume_sma_24[i]) or np.isnan(atr_ratio_1d[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 2.0x 24-period average
        vol_confirm = volume_current > 2.0 * volume_sma_24[i]
        
        # Volatility regime filter: trade only when volatility is elevated (ATR ratio > 1.0)
        vol_regime = atr_ratio_1d[i] > 1.0
        
        # Entry conditions - only trade extreme breaks for higher conviction
        enter_long = False
        enter_short = False
        
        # Long: Price breaks above Camarilla H4 level (strong breakout)
        if price_close > camarilla_H4_aligned[i] and vol_confirm and vol_regime:
            enter_long = True
        
        # Short: Price breaks below Camarilla L4 level (strong breakdown)
        if price_close < camarilla_L4_aligned[i] and vol_confirm and vol_regime:
            enter_short = True
        
        # Exit conditions: price returns to mean (Camarilla H3/L3 levels)
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns below H3 level (mean reversion)
            exit_long = price_close < camarilla_H3_aligned[i]
        elif position == -1:
            # Exit short if price returns above L3 level (mean reversion)
            exit_short = price_close > camarilla_L3_aligned[i]
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.20
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals