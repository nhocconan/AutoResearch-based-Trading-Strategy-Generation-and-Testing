#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_keltner_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
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
    
    # Calculate 1d ATR for Keltner channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_20_1d = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1d EMA (20-period) for Keltner center line
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Keltner upper and lower bands (2.0 * ATR)
    keltner_upper_1d = ema_20_1d + 2.0 * atr_20_1d
    keltner_lower_1d = ema_20_1d - 2.0 * atr_20_1d
    
    # Align Keltner bands to 4h timeframe
    keltner_upper_1d_aligned = align_htf_to_ltf(prices, df_1d, keltner_upper_1d)
    keltner_lower_1d_aligned = align_htf_to_ltf(prices, df_1d, keltner_lower_1d)
    
    # Volume confirmation: 20-period average on 4h
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volatility filter: ATR ratio (current ATR vs 20-period average)
    atr_ratio_1d = np.where(atr_20_1d[1:] > 0, atr_20_1d[1:] / np.roll(atr_20_1d, 1)[1:], 1.0)
    atr_ratio_1d = np.concatenate([[np.nan], atr_ratio_1d])
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    for i in range(200, n):
        # Skip if any required data is invalid
        if (np.isnan(keltner_upper_1d_aligned[i]) or np.isnan(keltner_lower_1d_aligned[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(atr_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Volatility filter: trade only when ATR ratio > 1.2 (expanding volatility)
        vol_expanding = atr_ratio_1d_aligned[i] > 1.2
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price breaks above Keltner upper band + volume confirmation + volatility expanding
        price_above_upper = price_close > keltner_upper_1d_aligned[i]
        if price_above_upper and vol_confirm and vol_expanding:
            enter_long = True
        
        # Short: Price breaks below Keltner lower band + volume confirmation + volatility expanding
        price_below_lower = price_close < keltner_lower_1d_aligned[i]
        if price_below_lower and vol_confirm and vol_expanding:
            enter_short = True
        
        # Exit conditions: price returns to the EMA (center line)
        ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
        exit_long = price_close < ema_20_1d_aligned[i]
        exit_short = price_close > ema_20_1d_aligned[i]
        
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

# Hypothesis: Keltner breakout on daily timeframe with volume confirmation and volatility expansion filter.
# Uses 1d Keltner channels (EMA20 ± 2*ATR20) for entry and EMA20 for exit.
# Volume confirmation (>1.5x 20-period average) ensures institutional participation.
# Volatility filter (ATR ratio > 1.2) ensures we only trade during expanding volatility phases.
# Works in both bull and breakout scenarios by capturing volatility expansion breakouts.
# Reduced position size to 0.25 to manage risk. Target: 20-40 trades/year to minimize fee drag.