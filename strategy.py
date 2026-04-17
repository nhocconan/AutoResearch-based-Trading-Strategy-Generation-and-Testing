#!/usr/bin/env python3
"""
4h Camarilla Pivot Breakout with Volume Spike and Choppiness Regime Filter
Long: Price breaks above Camarilla H3 level + volume > 2.0x 4h volume MA(20) + CHOP(14) > 61.8 (range)
Short: Price breaks below Camarilla L3 level + volume > 2.0x 4h volume MA(20) + CHOP(14) > 61.8 (range)
Exit: Price re-enters between H3 and L3 levels
Uses 1d Camarilla levels for structure and 4h volume/chop for confirmation and regime
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
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's OHLC
    # H4 = C + (H-L) * 1.500, H3 = C + (H-L) * 1.250, H2 = C + (H-L) * 1.166, H1 = C + (H-L) * 1.083
    # L1 = C - (H-L) * 1.083, L2 = C - (H-L) * 1.166, L3 = C - (H-L) * 1.250, L4 = C - (H-L) * 1.500
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Shift by 1 to get previous day's levels
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Calculate Camarilla levels
    H3 = prev_close + (prev_high - prev_low) * 1.250
    L3 = prev_close - (prev_high - prev_low) * 1.250
    
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # 4h volume moving average (20-period for confirmation)
    df_4h = get_htf_data(prices, '4h')
    volume_ma_20 = pd.Series(df_4h['volume']).rolling(window=20, min_periods=20).mean()
    volume_ma_20_4h = align_htf_to_ltf(prices, df_4h, volume_ma_20.values)
    
    # Choppiness Index on 4h (14-period)
    # CHOP = 100 * log10(sum(ATR over n) / (n * (max(high) - min(low)))) / log10(n)
    atr_list = []
    for i in range(len(df_4h)):
        if i < 1:
            atr_list.append(np.nan)
        else:
            tr = max(
                df_4h['high'].iloc[i] - df_4h['low'].iloc[i],
                abs(df_4h['high'].iloc[i] - df_4h['close'].iloc[i-1]),
                abs(df_4h['low'].iloc[i] - df_4h['close'].iloc[i-1])
            )
            atr_list.append(tr)
    
    atr_series = pd.Series(atr_list)
    atr_sum = atr_series.rolling(window=14, min_periods=14).sum()
    max_high = df_4h['high'].rolling(window=14, min_periods=14).max()
    min_low = df_4h['low'].rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (14 * (max_high - min_low))) / np.log10(14)
    chop_4h = align_htf_to_ltf(prices, df_4h, chop.values)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(volume_ma_20_4h[i]) or np.isnan(chop_4h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20_4h[i]
        chop_val = chop_4h[i]
        
        # Only trade in ranging markets (CHOP > 61.8)
        if chop_val > 61.8:
            if position == 0:
                # Long: break above H3 with volume spike
                if price > H3_aligned[i] and vol > 2.0 * vol_ma:
                    signals[i] = 0.25
                    position = 1
                # Short: break below L3 with volume spike
                elif price < L3_aligned[i] and vol > 2.0 * vol_ma:
                    signals[i] = -0.25
                    position = -1
            
            elif position == 1:
                # Long exit: price re-enters below H3
                if price < H3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            
            elif position == -1:
                # Short exit: price re-enters above L3
                if price > L3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # In trending markets, stay flat
            signals[i] = 0.0
            position = 0
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_Volume_Chop"
timeframe = "4h"
leverage = 1.0