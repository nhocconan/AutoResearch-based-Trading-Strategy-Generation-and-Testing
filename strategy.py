#!/usr/bin/env python3
"""
1d_WilliamsVixFix_VolumeSpike_ChopRegime_ATRStop
Hypothesis: Daily Williams Vix Fix (WVF) identifies extreme fear/greed reversals.
Enter long when WVF > 0.8 (extreme fear) with volume spike and choppy regime (CHOP > 61.8).
Enter short when WVF < 0.2 (extreme greed) with volume spike and choppy regime.
Exit on ATR(14) trailing stop (2.0*ATR) or WVF returning to neutral (0.4-0.6 range).
Designed for low trade frequency (<20 trades/year) to minimize fee drag.
Works in bull/bear via mean reversion from extremes in choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for regime filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === Williams Vix Fix (WVF) on daily close/high/low ===
    # WVF = ((HighestClose(L) - Low) / (HighestClose(L) - LowestLow(L))) * 100
    # where L = 22 period lookback (approx 1 month)
    lookback = 22
    high_close = prices['close'].rolling(window=lookback, min_periods=lookback).max()
    lowest_low = prices['low'].rolling(window=lookback, min_periods=lookback).min()
    highest_close = high_close  # alias for clarity
    wvf = ((highest_close - prices['low']) / (highest_close - lowest_low)) * 100
    wvf = wvf.values  # convert to numpy array
    
    # === Choppiness Index (CHOP) on 1w timeframe for regime filter ===
    # CHOP = 100 * log10(sum(ATR(1)) / (n * log(n))) / log10(n)
    # where ATR(1) = True Range, n = 14 period
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range for 1w
    tr1 = pd.Series(high_1w - low_1w)
    tr2 = pd.Series(np.abs(high_1w - np.roll(close_1w, 1)))
    tr3 = pd.Series(np.abs(low_1w - np.roll(close_1w, 1)))
    tr_1w = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = tr_1w.rolling(window=1, min_periods=1).mean().values  # ATR(1) = TR
    
    # Sum of ATR(1) over 14 periods
    sum_tr_14 = pd.Series(atr_1w).rolling(window=14, min_periods=14).sum().values
    n_val = 14
    chop = 100 * np.log10(sum_tr_14 / (n_val * np.log10(n_val))) / np.log10(n_val)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # === ATR (14-period) for stoploss on 1d timeframe ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(wvf[i]) or np.isnan(chop_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume spike: current volume > 1.5x 20-period average
            volume = prices['volume'].values
            vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
            vol_spike = volume[i] > 1.5 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
            
            # Regime filter: choppy market (CHOP > 61.8) for mean reversion
            choppy_regime = chop_aligned[i] > 61.8
            
            # Long conditions: extreme fear (WVF > 80) with volume spike in choppy market
            long_signal = (wvf[i] > 80.0) and vol_spike and choppy_regime
            
            # Short conditions: extreme greed (WVF < 20) with volume spike in choppy market
            short_signal = (wvf[i] < 20.0) and vol_spike and choppy_regime
            
            # Entry logic
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit when WVF returns to neutral range (40-60) or extreme greed (<20)
            elif wvf[i] < 60.0:  # returning from extreme fear
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit when WVF returns to neutral range (40-60) or extreme fear (>80)
            elif wvf[i] > 40.0:  # returning from extreme greed
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsVixFix_VolumeSpike_ChopRegime_ATRStop"
timeframe = "1d"
leverage = 1.0