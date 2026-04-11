#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prrices)
    if n < 200:
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
    
    # Calculate Camarilla levels on weekly data (using previous weekly bar's range)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    prev_close_1w[0] = np.nan
    
    camarilla_H4 = prev_close_1w + 1.1 * (prev_high_1w - prev_low_1w) / 2
    camarilla_L4 = prev_close_1w - 1.1 * (prev_high_1w - prev_low_1w) / 2
    
    # Align Camarilla levels to 1d timeframe
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_L4)
    
    # Volume confirmation: 20-period average on 1d
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(200, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_H4_aligned[i]) or np.isnan(camarilla_L4_aligned[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(atr_ratio_1w[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.8x 20-period average
        vol_confirm = volume_current > 1.8 * volume_sma_20[i]
        
        # Volatility regime filter: trade only when volatility is elevated (ATR ratio > 0.7)
        vol_regime = atr_ratio_1w[i] > 0.7
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price breaks above Camarilla H4 level + volume confirmation + volatility regime
        price_above_H4 = price_close > camarilla_H4_aligned[i]
        if price_above_H4 and vol_confirm and vol_regime:
            enter_long = True
        
        # Short: Price breaks below Camarilla L4 level + volume confirmation + volatility regime
        price_below_L4 = price_close < camarilla_L4_aligned[i]
        if price_below_L4 and vol_confirm and vol_regime:
            enter_short = True
        
        # Exit conditions: price crosses back through the Camarilla mid-point (C level)
        exit_long = False
        exit_short = False
        
        # Calculate Camarilla C level (close of previous weekly bar)
        camarilla_C = prev_close_1w
        camarilla_C_aligned = align_htf_to_ltf(prices, df_1w, camarilla_C)
        
        if position == 1:
            # Exit long if price crosses below Camarilla C level
            exit_long = price_close < camarilla_C_aligned[i]
        elif position == -1:
            # Exit short if price crosses above Camarilla C level
            exit_short = price_close > camarilla_C_aligned[i]
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.30
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.30
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.30 if position == 1 else (-0.30 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Weekly Camarilla breakout with volume and volatility regime filters on daily chart.
# Uses weekly Camarilla levels (H4/L4) for entry and C level for exit on daily timeframe.
# Volume confirmation (>1.8x 20-period average) ensures institutional participation.
# Volatility regime filter (ATR ratio > 0.7) avoids low-volatility chop.
# Works in both bull and bear markets by capturing breakouts from key weekly levels.
# Target: 10-25 trades/year to minimize fee drag on 1d timeframe.