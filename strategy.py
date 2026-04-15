#!/usr/bin/env python3
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
    
    # Get weekly HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Calculate weekly Donchian channels (20-period) for trend
    highest_20 = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Align HTF indicators to daily timeframe with proper delay
    highest_20_aligned = align_htf_to_ltf(prices, df_1w, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1w, lowest_20)
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.concatenate([[close[0]], close[:-1]])))
    tr3 = pd.Series(np.abs(low - np.concatenate([[close[0]], close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate daily volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or 
            np.isnan(atr_14[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Daily price breaks above weekly Donchian high with volume confirmation → long
        # 2. Daily price breaks below weekly Donchian low with volume confirmation → short
        # 3. Volatility filter: ATR > 0.3% of price (avoid low volatility chop)
        # 4. Volume confirmation: volume > 1.5x average
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: daily breakout above weekly high
        if (close[i] > highest_20_aligned[i] and            # daily price above weekly Donchian high
            volume_ratio[i] > 1.5 and                      # Volume confirmation
            atr_14[i] > 0.003 * close[i]):                 # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: daily breakdown below weekly low
        elif (close[i] < lowest_20_aligned[i] and          # daily price below weekly Donchian low
              volume_ratio[i] > 1.5 and                    # Volume confirmation
              atr_14[i] > 0.003 * close[i]):               # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyDonchian20_Breakout_Volume_ATR_Filter"
timeframe = "1d"
leverage = 1.0