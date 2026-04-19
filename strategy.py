#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter with 12h EMA trend and volume confirmation.
# Long when EMA(34,12h) > EMA(89,12h) (uptrend) AND Choppiness Index(14) < 38.2 (trending market) AND volume > 1.5x 20-period average
# Short when EMA(34,12h) < EMA(89,12h) (downtrend) AND Choppiness Index(14) < 38.2 AND volume > 1.5x 20-period average
# Exit when EMA crossover reverses OR Choppiness Index > 61.8 (choppy market)
# Uses trend following in trending markets only, avoiding whipsaws in chop.
# Target: 20-30 trades/year per symbol.

name = "4h_EMA_Chop_Volume_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend and Choppiness Index
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate EMA(34) and EMA(89) on 12h close
    close_12h = df_12h['close'].values
    ema34 = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89 = pd.Series(close_12h).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align EMA arrays to 4h timeframe
    ema34_aligned = align_htf_to_ltf(prices, df_12h, ema34)
    ema89_aligned = align_htf_to_ltf(prices, df_12h, ema89)
    
    # Calculate Choppiness Index on 12h data
    # True Range
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = np.abs(df_12h['high'] - df_12h['close'].shift(1))
    tr3 = np.abs(df_12h['low'] - df_12h['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(df_12h['high']).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(df_12h['low']).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(tr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero
    range_hl = hh - ll
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)  # small value to avoid div by zero
    chop = 100 * np.log10(tr_sum / range_hl) / np.log10(14)
    
    # Align Choppiness Index to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # Get 20-period volume average for confirmation (using 4h data directly)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(89, 20)  # Ensure EMA89 and vol MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_aligned[i]) or np.isnan(ema89_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema34_val = ema34_aligned[i]
        ema89_val = ema89_aligned[i]
        chop_val = chop_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Trend condition: EMA34 > EMA89 for uptrend, < for downtrend
        uptrend = ema34_val > ema89_val
        downtrend = ema34_val < ema89_val
        
        # Choppiness regime: < 38.2 = trending, > 61.8 = choppy
        trending_market = chop_val < 38.2
        choppy_market = chop_val > 61.8
        
        # Volume confirmation
        vol_confirm = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: uptrend + trending market + volume confirmation
            if uptrend and trending_market and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + trending market + volume confirmation
            elif downtrend and trending_market and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend reverses OR market becomes choppy
            if not uptrend or choppy_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reverses OR market becomes choppy
            if not downtrend or choppy_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals