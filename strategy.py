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
    weekly_volume = df_1w['volume'].values
    
    # Calculate weekly ATR(14) for volatility filter
    tr1 = pd.Series(weekly_high - weekly_low)
    tr2 = pd.Series(np.abs(weekly_high - np.concatenate([[weekly_close[0]], weekly_close[:-1]])))
    tr3 = pd.Series(np.abs(weekly_low - np.concatenate([[weekly_close[0]], weekly_close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate weekly EMA(34) for trend filter
    ema_34 = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 1d timeframe with proper delay
    atr_14_1d = align_htf_to_ltf(prices, df_1w, atr_14)
    ema_34_1d = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Calculate daily Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_1d[i]) or np.isnan(ema_34_1d[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 1d price breaks above Donchian(20) high with volume confirmation → long (only in weekly uptrend)
        # 2. 1d price breaks below Donchian(20) low with volume confirmation → short (only in weekly downtrend)
        # 3. Volatility filter: ATR > 0.3% of price (avoid low volatility chop)
        # 4. Volume confirmation: volume > 1.5x average
        # 5. Weekly trend filter: price above/below EMA34
        # 6. Discrete position sizing: 0.25
        
        # Long conditions: 1d breakout above Donchian high in weekly uptrend
        if (close[i] > highest_20[i] and            # 1d price above Donchian high
            close[i] > ema_34_1d[i] and             # Weekly uptrend filter
            volume_ratio[i] > 1.5 and               # Volume confirmation
            atr_14_1d[i] > 0.003 * close[i]):       # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: 1d breakdown below Donchian low in weekly downtrend
        elif (close[i] < lowest_20[i] and           # 1d price below Donchian low
              close[i] < ema_34_1d[i] and           # Weekly downtrend filter
              volume_ratio[i] > 1.5 and             # Volume confirmation
              atr_14_1d[i] > 0.003 * close[i]):     # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyDonchian20_Breakout_Volume_ATR_Filter"
timeframe = "1d"
leverage = 1.0