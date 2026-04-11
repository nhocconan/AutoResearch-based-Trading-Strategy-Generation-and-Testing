#!/usr/bin/env python3
"""
4h_1d_camarilla_breakout_volume_trend_v2
Strategy: 4h Camarilla pivot breakout with 1d volume confirmation and trend filter
Timeframe: 4h
Leverage: 1.0
Hypothesis: Uses 4h price breakout above/below Camarilla pivot levels (H3/L3) calculated from previous 1d candle, confirmed by 1d volume spike (>1.5x 20-period average) and 1d EMA50 trend filter. Designed to capture breakouts in trending markets while avoiding false breakouts in chop. Works in both bull/bear markets by following 1d trend direction. Target: 20-50 trades per year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_hlf

name = "4h_1d_camarilla_breakout_volume_trend_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prrices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d close
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Camarilla levels: H3/L3 = close ± 1.1*(high-low)/2
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 1d volume confirmation: volume > 1.5x 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = vol_1d > (1.5 * vol_ma_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        
        # Trend filter: price above/below EMA50
        uptrend = price_close > ema_50_1d_aligned[i]
        downtrend = price_close < ema_50_1d_aligned[i]
        
        # Breakout conditions
        breakout_long = price_high > camarilla_h3_aligned[i] and vol_spike_aligned[i]
        breakout_short = price_low < camarilla_l3_aligned[i] and vol_spike_aligned[i]
        
        # Entry logic: follow trend direction
        long_signal = breakout_long and uptrend
        short_signal = breakout_short and downtrend
        
        # Exit when price returns to pivot level (close price)
        pivot_1d = close_1d
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
        exit_long = position == 1 and price_close < pivot_aligned[i]
        exit_short = position == -1 and price_close > pivot_aligned[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Uses 4h price breakout above/below Camarilla pivot levels (H3/L3) calculated from previous 1d candle, confirmed by 1d volume spike (>1.5x 20-period average) and 1d EMA50 trend filter. Designed to capture breakouts in trending markets while avoiding false breakouts in chop. Works in both bull/bear markets by following 1d trend direction. Target: 20-50 trades per year.