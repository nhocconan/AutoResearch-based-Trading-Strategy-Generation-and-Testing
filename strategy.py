#!/usr/bin/env python3
"""
12h 1D Volatility-Adjusted Breakout with Volume and Regime Filter
Long: Price breaks above prior 1D high + volume > 1.5x 12h volume MA + price > 1D EMA50 + Choppiness < 50
Short: Price breaks below prior 1D low + volume > 1.5x 12h volume MA + price < 1D EMA50 + Choppiness < 50
Exit: Opposite break of prior 1D level
Volatility filter: Skip if 12h ATR > 1.5x 12h ATR MA (50) to avoid whipsaw in high volatility
Target: 25-35 trades/year per symbol
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
    
    # Get 12h data for volume and volatility filters
    df_12h = get_htf_data(prices, '12h')
    volume_ma_24 = pd.Series(df_12h['volume']).rolling(window=24, min_periods=24).mean()
    atr_12h = pd.Series(np.maximum.reduce([
        df_12h['high'] - df_12h['low'],
        np.abs(df_12h['high'] - df_12h['close'].shift(1)),
        np.abs(df_12h['low'] - df_12h['close'].shift(1))
    ])).rolling(window=14, min_periods=14).mean()
    atr_ma_50 = pd.Series(atr_12h).rolling(window=50, min_periods=50).mean()
    atr_ratio = atr_12h / atr_ma_50
    
    volume_ma_24_12h = align_htf_to_ltf(prices, df_12h, volume_ma_24.values)
    atr_ratio_12h = align_htf_to_ltf(prices, df_12h, atr_ratio.values)
    
    # 12h Choppiness index for regime filter
    chop_atr_sum = pd.Series(atr_12h).rolling(window=14, min_periods=14).sum()
    chop_high_low = pd.Series(df_12h['high'].rolling(window=14, min_periods=14).max() - 
                              df_12h['low'].rolling(window=14, min_periods=14).min())
    chop = 100 * np.log10(chop_atr_sum / chop_high_low) / np.log10(14)
    chop_12h = align_htf_to_ltf(prices, df_12h, chop.values)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(prior_1d_high_aligned[i]) or np.isnan(prior_1d_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_24_12h[i]) or
            np.isnan(atr_ratio_12h[i]) or np.isnan(chop_12h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_24_12h[i]
        atr_ratio_val = atr_ratio_12h[i]
        chop_val = chop_12h[i]
        
        # Volatility filter: avoid high volatility periods
        vol_filter = atr_ratio_val < 1.5
        
        # Regime filter: only trade in non-choppy markets
        regime_filter = chop_val < 50
        
        if position == 0:
            # Long: break above prior 1D high + volume + 1D trend + filters
            if (price > prior_1d_high_aligned[i] and vol > 1.5 * vol_ma and 
                price > ema_50_1d_aligned[i] and vol_filter and regime_filter):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below prior 1D low + volume + 1D trend + filters
            elif (price < prior_1d_low_aligned[i] and vol > 1.5 * vol_ma and 
                  price < ema_50_1d_aligned[i] and vol_filter and regime_filter):
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

name = "12h_VolAdjusted_Breakout_Volume_EMAFilter"
timeframe = "12h"
leverage = 1.0