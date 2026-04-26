#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dATR_Volume_Confluence_v1
Hypothesis: Trade Camarilla R1/S1 breakouts with 1d ATR trend filter (price > 1d ATR above/below EMA20) and volume spike confirmation.
Uses ATR trailing stop (2.5x) for risk control. Designed for low trade frequency (<400 total) to minimize fee drag and improve generalization.
Works in bull via breakouts with trend, in bear via mean-reversion fades at extremes when volatility spikes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA(20) for trend reference
    ema_20_1d = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    # 1d ATR(14) for volatility normalization
    tr1 = df_1d['high'].values[1:] - df_1d['low'].values[1:]
    tr2 = np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1])
    tr3 = np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous 1d bar
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align HTF indicators to 4h timeframe
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: 2.0x median volume (balanced frequency)
    vol_median = pd.Series(volume).rolling(window=24, min_periods=24).median().values
    
    # 4h ATR for stoploss
    tr1_4h = high[1:] - low[1:]
    tr2_4h = np.abs(high[1:] - close[:-1])
    tr3_4h = np.abs(low[1:] - close[:-1])
    tr_4h = np.concatenate([[np.nan], np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))])
    atr_14_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    bars_since_entry = 0
    
    # Warmup: max of 1d EMA (20), volume median (24), 4h ATR (14)
    start_idx = max(20, 24, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_median[i]) or 
            np.isnan(atr_14_4h[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_20_1d_val = ema_20_1d_aligned[i]
        atr_14_1d_val = atr_14_1d_aligned[i]
        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_14_4h_val = atr_14_4h[i]
        
        if position == 0:
            # Long: break above R1, price > 1d EMA20 + 0.5*ATR (uptrend), volume spike
            long_signal = (high_val > camarilla_r1_val) and \
                          (close_val > ema_20_1d_val + 0.5 * atr_14_1d_val) and \
                          (volume_val > 2.0 * vol_median_val)
            # Short: break below S1, price < 1d EMA20 - 0.5*ATR (downtrend), volume spike
            short_signal = (low_val < camarilla_s1_val) and \
                           (close_val < ema_20_1d_val - 0.5 * atr_14_1d_val) and \
                           (volume_val > 2.0 * vol_median_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                long_stop = entry_price - 2.5 * atr_14_4h_val
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                short_stop = entry_price + 2.5 * atr_14_4h_val
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long with minimum holding period
            bars_since_entry += 1
            signals[i] = 0.25
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 2.5 * atr_14_4h_val)
            # Exit: trailing stop hit or trend reversal (price < EMA20) after minimum holding period
            if bars_since_entry >= 6 and ((low_val < long_stop) or (close_val < ema_20_1d_val)):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short with minimum holding period
            bars_since_entry += 1
            signals[i] = -0.25
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.5 * atr_14_4h_val)
            # Exit: trailing stop hit or trend reversal (price > EMA20) after minimum holding period
            if bars_since_entry >= 6 and ((high_val > short_stop) or (close_val > ema_20_1d_val)):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dATR_Volume_Confluence_v1"
timeframe = "4h"
leverage = 1.0