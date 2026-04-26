#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_ATRStop_v2
Hypothesis: 4h breakout above/below daily Camarilla R1/S1 levels in direction of 12h EMA50 trend, confirmed by volume spike (>1.8x 20-bar MA) and ATR-based stoploss (2x ATR). Uses 12h HTF for trend alignment and Camarilla levels from daily OHLC for institutional support/resistance. Volume confirmation reduces false breakouts. ATR stoploss manages risk. Designed for 19-50 trades/year (75-200 total over 4 years) to avoid fee drag. Works in both bull and bear markets by following the 12h trend while using Camarilla structure for precise entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar (OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1, S1 levels (based on previous day's range)
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (no additional delay needed as they're based on completed 1d bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25  # Position size
    
    # Warmup: max of calculations (20 for vol, 50 for ema, 14 for atr)
    start_idx = max(20, 50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_12h_aligned[i]
        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Determine 12h trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_12h = close_val > ema_50_val
        bearish_12h = close_val < ema_50_val
        
        # Entry conditions: breakout of Camarilla level in trend direction with volume
        long_entry = (close_val > camarilla_r1_val) and bullish_12h and vol_spike
        short_entry = (close_val < camarilla_s1_val) and bearish_12h and vol_spike
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -base_size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - check exit conditions
            # Exit: stoploss hit OR opposite Camarilla level touch
            stoploss_long = entry_price - (2.0 * atr_val)
            exit_long = (close_val < stoploss_long) or (close_val < camarilla_s1_val)
            if exit_long:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = base_size
        elif position == -1:
            # Short - check exit conditions
            # Exit: stoploss hit OR opposite Camarilla level touch
            stoploss_short = entry_price + (2.0 * atr_val)
            exit_short = (close_val > stoploss_short) or (close_val > camarilla_r1_val)
            if exit_short:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_ATRStop_v2"
timeframe = "4h"
leverage = 1.0