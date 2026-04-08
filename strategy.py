#!/usr/bin/env python3
# 1h_4h1d_donchian_breakout_volume_v1
# Hypothesis: Trade Donchian breakouts on 1h with 4h/1d trend filter and volume confirmation.
# Long when price breaks 1h high with 4h/1d uptrend and volume surge.
# Short when price breaks 1h low with 4h/1d downtrend and volume surge.
# Uses 4h EMA20/50 and 1d EMA50/200 for trend, ATR for stop, volume > 1.5x average for confirmation.
# Designed for low trade frequency (15-30/year) to avoid fee drag in choppy 1h market.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_donchian_breakout_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h EMA for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d EMA for stronger trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # ATR for volatility and stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 200  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(high_20[i]) or np.isnan(low_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        # Trend conditions
        uptrend_4h = ema20_4h_aligned[i] > ema50_4h_aligned[i]
        downtrend_4h = ema20_4h_aligned[i] < ema50_4h_aligned[i]
        uptrend_1d = ema50_1d_aligned[i] > ema200_1d_aligned[i]
        downtrend_1d = ema50_1d_aligned[i] < ema200_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: trend reversal OR stoploss hit
            if not uptrend_4h or not uptrend_1d or close[i] < ema20_4h_aligned[i] - 2.0 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: trend reversal OR stoploss hit
            if not downtrend_4h or not downtrend_1d or close[i] > ema20_4h_aligned[i] + 2.0 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: break above 1h high with uptrend and volume surge
            if close[i] > high_20[i] and uptrend_4h and uptrend_1d and vol_surge:
                position = 1
                signals[i] = 0.20
            # Short entry: break below 1h low with downtrend and volume surge
            elif close[i] < low_20[i] and downtrend_4h and downtrend_1d and vol_surge:
                position = -1
                signals[i] = -0.20
    
    return signals