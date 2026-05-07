#!/usr/bin/env python3
"""
4h_Keltner_Breakout_1dTrend_Volume
Hypothesis: Keltner Channel breakout with 1d EMA trend filter and volume spike. 
Keltner Channels (ATR-based) adapt better to volatility than fixed bands, reducing false breakouts in choppy markets. 
Trades only in direction of daily trend (EMA34) to avoid counter-trend losses in bear markets. 
Volume spike (>2x 20-bar average) confirms institutional interest. 
Designed for low trade frequency (20-40/year) to minimize fee drag while capturing strong trending moves.
"""

name = "4h_Keltner_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(20) for Keltner Channels
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Keltner Channel: EMA(20) ± 2*ATR
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema_20 + 2 * atr
    lower_keltner = ema_20 - 2 * atr
    
    # Volume confirmation: 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Warmup for EMA34 and ATR
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get aligned daily close for trend determination
        daily_close_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
        if np.isnan(daily_close_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        daily_trend_up = daily_close_aligned[i] > ema_34_1d_aligned[i]
        daily_trend_down = daily_close_aligned[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Keltner, volume spike, price above daily EMA34
            if (close[i] > upper_keltner[i] and 
                vol_ratio[i] > 2.0 and 
                daily_trend_up):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Keltner, volume spike, price below daily EMA34
            elif (close[i] < lower_keltner[i] and 
                  vol_ratio[i] > 2.0 and 
                  daily_trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below EMA(20) or trend changes
            if close[i] < ema_20[i] or not daily_trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above EMA(20) or trend changes
            if close[i] > ema_20[i] or not daily_trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals