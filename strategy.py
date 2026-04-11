#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_volume_v1"
timeframe = "1d"
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return signals
    
    # Calculate weekly ATR for volatility filter (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_14_1w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Calculate weekly ATR MA (20-period) for volatility regime filter
    atr_ma_20_1w = pd.Series(atr_14_1w_aligned).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly ATR ratio (current / MA) for regime detection
    atr_ratio_1w = np.where(atr_ma_20_1w > 0, atr_14_1w_aligned / atr_ma_20_1w, 1.0)
    
    # Calculate Camarilla levels on daily data (using previous daily bar's range)
    prev_high_1d = np.roll(high, 1)
    prev_low_1d = np.roll(low, 1)
    prev_close_1d = np.roll(close, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_H4 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_L4 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 2
    
    # Volume confirmation: 20-period average on daily
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Price momentum filter: close above/below 20-period SMA on daily
    close_sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # Track consecutive triggers to prevent whipsaw
    consecutive_long_trigger = 0
    consecutive_short_trigger = 0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_H4[i]) or np.isnan(camarilla_L4[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(atr_ratio_1w[i]) or
            np.isnan(close_sma_20[i])):
            signals[i] = 0.0
            consecutive_long_trigger = 0
            consecutive_short_trigger = 0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Volatility regime filter: trade only when volatility is elevated (ATR ratio > 0.8)
        vol_regime = atr_ratio_1w[i] > 0.8
        
        # Price momentum filter: price above SMA for longs, below SMA for shorts
        mom_filter_long = price_close > close_sma_20[i]
        mom_filter_short = price_close < close_sma_20[i]
        
        # Entry conditions with consecutive trigger requirement
        enter_long = False
        enter_short = False
        
        # Long: Price breaks above Camarilla H4 level + volume confirmation + volatility regime + momentum filter
        price_above_H4 = price_close > camarilla_H4[i]
        if price_above_H4 and vol_confirm and vol_regime and mom_filter_long:
            consecutive_long_trigger += 1
            consecutive_short_trigger = 0
            if consecutive_long_trigger >= 2:  # Require 2 consecutive triggers
                enter_long = True
        else:
            consecutive_long_trigger = 0
        
        # Short: Price breaks below Camarilla L4 level + volume confirmation + volatility regime + momentum filter
        price_below_L4 = price_close < camarilla_L4[i]
        if price_below_L4 and vol_confirm and vol_regime and mom_filter_short:
            consecutive_short_trigger += 1
            consecutive_long_trigger = 0
            if consecutive_short_trigger >= 2:  # Require 2 consecutive triggers
                enter_short = True
        else:
            consecutive_short_trigger = 0
        
        # Exit conditions: price crosses back through the Camarilla mid-point (C level)
        exit_long = False
        exit_short = False
        
        # Calculate Camarilla C level (close of previous daily bar)
        camarilla_C = prev_close_1d
        
        if position == 1:
            # Exit long if price crosses below Camarilla C level
            exit_long = price_close < camarilla_C[i]
        elif position == -1:
            # Exit short if price crosses above Camarilla C level
            exit_short = price_close > camarilla_C[i]
        
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