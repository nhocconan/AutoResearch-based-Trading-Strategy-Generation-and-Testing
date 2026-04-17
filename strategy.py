#!/usr/bin/env python3
"""
6h 1D Prior Day Range Breakout with Volume and 1D Trend Filter + Volatility Filter
Long: Price breaks above prior 1D high + volume > 1.5x 6h volume MA + price > 1D EMA50 + volatility expansion (ATR6h > 1.2 * ATR12h)
Short: Price breaks below prior 1D low + volume > 1.5x 6h volume MA + price < 1D EMA50 + volatility expansion
Exit: Opposite break of prior 1D level
Adds volatility filter to avoid false breakouts in low volatility chop, improves signal quality.
Target: 15-25 trades/year per symbol
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
    
    # Get 1D data for prior high/low and trend filter
    df_1d = get_htf_data(prices, '1d')
    prior_1d_high = df_1d['high'].shift(1)  # Prior day's high
    prior_1d_low = df_1d['low'].shift(1)    # Prior day's low
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    prior_1d_high_aligned = align_htf_to_ltf(prices, df_1d, prior_1d_high.values)
    prior_1d_low_aligned = align_htf_to_ltf(prices, df_1d, prior_1d_low.values)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h volume moving average (4-period for confirmation)
    df_6h = get_htf_data(prices, '6h')
    volume_ma_4 = pd.Series(df_6h['volume']).rolling(window=4, min_periods=4).mean()
    volume_ma_4_6h = align_htf_to_ltf(prices, df_6h, volume_ma_4.values)
    
    # Volatility filter: ATR(6h) > 1.2 * ATR(12h) - volatility expansion
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high']
    low_12h = df_12h['low']
    close_12h = df_12h['close']
    
    # Calculate ATR for 6h and 12h
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr1_12h[0] = np.nan
    tr2_12h[0] = np.nan
    tr3_12h[0] = np.nan
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Volatility expansion condition
    vol_expansion = atr_6h > (1.2 * atr_12h_aligned)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(prior_1d_high_aligned[i]) or np.isnan(prior_1d_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_4_6h[i]) or
            np.isnan(vol_expansion[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_4_6h[i]
        
        if position == 0:
            # Long: break above prior 1D high + volume + 1D trend + volatility expansion
            if price > prior_1d_high_aligned[i] and vol > 1.5 * vol_ma and price > ema_50_1d_aligned[i] and vol_expansion[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below prior 1D low + volume + 1D trend + volatility expansion
            elif price < prior_1d_low_aligned[i] and vol > 1.5 * vol_ma and price < ema_50_1d_aligned[i] and vol_expansion[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: break below prior 1D low
            if price < prior_1d_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above prior 1D high
            if price > prior_1d_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Prior1D_HL_Breakout_Volume_Trend_VolFilter"
timeframe = "6h"
leverage = 1.0