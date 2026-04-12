#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v26"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from daily data (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4/L4 (resistance/support)
    # Formula: H4 = C + 1.5*(H-L), L4 = C - 1.5*(H-L)
    camarilla_H4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_L4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align to 4h timeframe
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    
    # Volume confirmation: current volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    # Choppiness regime filter (from daily)
    # Chop > 61.8 = ranging (mean revert), Chop < 38.2 = trending (trend follow)
    hl14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    atr14 = pd.Series(hl14 - ll14).rolling(window=14, min_periods=14).mean().values
    sum_true_range = pd.Series(hl14 - ll14).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_true_range / atr14) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Market regime: chop < 50 = trending, chop >= 50 = ranging
    trending_market = chop_aligned < 50
    ranging_market = chop_aligned >= 50
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # warmup for chop calculation
        # Skip if not ready
        if (np.isnan(camarilla_H4_aligned[i]) or np.isnan(camarilla_L4_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Price levels
        price = close[i]
        H4 = camarilla_H4_aligned[i]
        L4 = camarilla_L4_aligned[i]
        vol_ok = volume_ok[i]
        
        # In trending markets: breakout strategy
        # In ranging markets: mean reversion at extremes
        if trending_market[i]:
            # Trending: breakout of H4/L4 with volume
            long_signal = (price > H4) and vol_ok
            short_signal = (price < L4) and vol_ok
            
            # Exit when price returns to midpoint
            midpoint = (H4 + L4) / 2
            exit_long = price < midpoint
            exit_short = price > midpoint
            
        else:
            # Ranging: mean reversion at H4/L4
            long_signal = (price < L4) and vol_ok  # buy at support
            short_signal = (price > H4) and vol_ok  # sell at resistance
            
            # Exit when price returns to midpoint
            midpoint = (H4 + L4) / 2
            exit_long = price > midpoint
            exit_short = price < midpoint
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals