#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_breakout_volume_v3"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return signals
    
    # Calculate 12h ATR for volatility filter (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_14_12h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    
    # Calculate 12h ATR MA (20-period) for volatility regime filter
    atr_ma_20_12h = pd.Series(atr_14_12h_aligned).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h ATR ratio (current / MA) for regime detection
    atr_ratio_12h = np.where(atr_ma_20_12h > 0, atr_14_12h_aligned / atr_ma_20_12h, 1.0)
    
    # Calculate 12h ATR MA (50-period) for trend filter
    atr_ma_50_12h = pd.Series(atr_14_12h_aligned).rolling(window=50, min_periods=50).mean().values
    
    # Calculate Camarilla levels on 12h data (using previous 12h bar's range)
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_close_12h = np.roll(close_12h, 1)
    prev_high_12h[0] = np.nan
    prev_low_12h[0] = np.nan
    prev_close_12h[0] = np.nan
    
    camarilla_H4_12h = prev_close_12h + 1.1 * (prev_high_12h - prev_low_12h) / 2
    camarilla_L4_12h = prev_close_12h - 1.1 * (prev_high_12h - prev_low_12h) / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_H4_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_H4_12h)
    camarilla_L4_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_L4_12h)
    
    # Volume confirmation: 20-period average on 4h
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(200, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_H4_12h_aligned[i]) or np.isnan(camarilla_L4_12h_aligned[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(atr_ratio_12h[i]) or np.isnan(atr_ma_50_12h[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average (stricter filter)
        vol_confirm = volume_current > 2.0 * volume_sma_20[i]
        
        # Trend filter: trade only when ATR is above its 50-period MA (trending market)
        trending = atr_14_12h_aligned[i] > atr_ma_50_12h[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price breaks above Camarilla H4 level + volume confirmation + trending
        price_above_H4 = price_close > camarilla_H4_12h_aligned[i]
        if price_above_H4 and vol_confirm and trending:
            enter_long = True
        
        # Short: Price breaks below Camarilla L4 level + volume confirmation + trending
        price_below_L4 = price_close < camarilla_L4_12h_aligned[i]
        if price_below_L4 and vol_confirm and trending:
            enter_short = True
        
        # Exit conditions: price crosses back through the Camarilla mid-point (C level)
        exit_long = False
        exit_short = False
        
        # Calculate Camarilla C level (close of previous 12h bar)
        camarilla_C_12h = prev_close_12h
        camarilla_C_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_C_12h)
        
        if position == 1:
            # Exit long if price crosses below Camarilla C level
            exit_long = price_close < camarilla_C_12h_aligned[i]
        elif position == -1:
            # Exit short if price crosses above Camarilla C level
            exit_short = price_close > camarilla_C_12h_aligned[i]
        
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

# Hypothesis: Camarilla breakout on 12h timeframe with stricter volume confirmation (2.0x) and ATR trend filter.
# Uses 12h Camarilla levels (H4/L4) for entry and C level for exit.
# Volume confirmation (>2.0x 20-period average) ensures strong institutional participation.
# ATR trend filter (ATR > 50-period MA) ensures we only trade in trending markets.
# Reduced position size to 0.25 to manage risk. Target: 15-30 trades/year to minimize fee drag.