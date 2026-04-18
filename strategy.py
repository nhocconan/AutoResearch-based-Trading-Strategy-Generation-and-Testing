#!/usr/bin/env python3
"""
12h_1d_Keltner_Channel_Breakout_Volume_Trend
Hypothesis: Breakout of daily Keltner Channel (upper/lower bands) with volume confirmation and 12h trend bias.
Trades only in the direction of the 12h EMA trend to avoid whipsaws in choppy markets.
Targets 12-37 trades per year by using Keltner Channel (ATR-based) bands, volume confirmation, and trend filter.
Works in both bull and bear markets by following the 12h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Keltner Channel (upper/lower bands)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA (20) for Keltner Channel middle line
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1d ATR (10) for Keltner Channel width
    tr_1d = np.maximum(
        df_1d['high'].values - df_1d['low'].values,
        np.maximum(
            np.abs(df_1d['high'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]])),
            np.abs(df_1d['low'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]]))
        )
    )
    atr_10_1d = pd.Series(tr_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 1d Keltner Channel upper and lower bands
    keltner_upper_1d = ema_20_1d + (2.0 * atr_10_1d)
    keltner_lower_1d = ema_20_1d - (2.0 * atr_10_1d)
    
    # Calculate 1d EMA (20) for trend bias
    ema_20_1d_for_trend = ema_20_1d  # reuse calculated EMA
    
    # Align all levels to 12h timeframe (wait for bar close)
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1d, keltner_upper_1d)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1d, keltner_lower_1d)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d_for_trend)
    
    # Get 12h trend (EMA34) for directional bias
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or 
            np.isnan(ema_20_1d_aligned[i]) or np.isnan(ema_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above 1d Keltner upper band, above 1d EMA, with volume, and 12h uptrend
            if (close[i] > keltner_upper_aligned[i] and 
                close[i] > ema_20_1d_aligned[i] and vol_confirm[i] and 
                close[i] > ema_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below 1d Keltner lower band, below 1d EMA, with volume, and 12h downtrend
            elif (close[i] < keltner_lower_aligned[i] and 
                  close[i] < ema_20_1d_aligned[i] and vol_confirm[i] and 
                  close[i] < ema_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns to 1d EMA or 12h downtrend
            if (not np.isnan(ema_20_1d_aligned[i]) and close[i] < ema_20_1d_aligned[i]) or \
               (close[i] < ema_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to 1d EMA or 12h uptrend
            if (not np.isnan(ema_20_1d_aligned[i]) and close[i] > ema_20_1d_aligned[i]) or \
               (close[i] > ema_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Keltner_Channel_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0