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
    
    # Get daily HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_volume = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels for daily timeframe
    # Camarilla: based on previous day's range, more responsive than standard pivots
    daily_range = daily_high - daily_low
    camarilla_h4 = daily_close + daily_range * 1.1 / 2
    camarilla_h3 = daily_close + daily_range * 1.1 / 4
    camarilla_h2 = daily_close + daily_range * 1.1 / 6
    camarilla_h1 = daily_close + daily_range * 1.1 / 12
    camarilla_l1 = daily_close - daily_range * 1.1 / 12
    camarilla_l2 = daily_close - daily_range * 1.1 / 6
    camarilla_l3 = daily_close - daily_range * 1.1 / 4
    camarilla_l4 = daily_close - daily_range * 1.1 / 2
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = pd.Series(daily_high - daily_low)
    tr2 = pd.Series(np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr3 = pd.Series(np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 12h timeframe with proper delay
    camarilla_h4_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_h2_12h = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    camarilla_h1_12h = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    camarilla_l1_12h = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    camarilla_l2_12h = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    camarilla_l3_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    atr_14_12h = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 12h Donchian channels (20-period) for breakout confirmation
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Calculate 12h ADX(14) for regime filter
    plus_dm = pd.Series(np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                                 np.maximum(high[1:] - high[:-1], 0), 0))
    minus_dm = pd.Series(np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                                  np.maximum(low[:-1] - low[1:], 0), 0))
    plus_dm = pd.Series(np.concatenate([[0], plus_dm.values]))
    minus_dm = pd.Series(np.concatenate([[0], minus_dm.values]))
    
    tr_latest = pd.Series(np.maximum.reduce([
        high - low,
        np.abs(high - np.concatenate([[close[0]], close[:-1]])),
        np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    ]))
    atr_latest = tr_latest.ewm(span=14, adjust=False, min_periods=14).mean()
    
    plus_di = 100 * (plus_dm.ewm(span=14, adjust=False, min_periods=14).mean() / atr_latest)
    minus_di = 100 * (minus_dm.ewm(span=14, adjust=False, min_periods=14).mean() / atr_latest)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
    adx = dx.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_h4_12h[i]) or np.isnan(camarilla_l4_12h[i]) or 
            np.isnan(atr_14_12h[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 12h price breaks above Camarilla H4 with volume confirmation → long
        # 2. 12h price breaks below Camarilla L4 with volume confirmation → short
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Regime filter: ADX > 25 (trending market)
        # 5. Volatility filter: ATR > 0.3% of price (avoid extremely low volatility)
        # 6. Discrete position sizing: 0.25
        
        # Long conditions: 12h breakout above Camarilla H4
        if (close[i] > camarilla_h4_12h[i] and            # 12h price above H4
            close[i] > highest_20[i] and                  # Also above Donchian high for confirmation
            volume_ratio[i] > 1.5 and                     # Volume confirmation
            adx[i] > 25 and                               # Trending regime
            atr_14_12h[i] > 0.003 * close[i]):            # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: 12h breakdown below Camarilla L4
        elif (close[i] < camarilla_l4_12h[i] and          # 12h price below L4
              close[i] < lowest_20[i] and                 # Also below Donchian low for confirmation
              volume_ratio[i] > 1.5 and                   # Volume confirmation
              adx[i] > 25 and                             # Trending regime
              atr_14_12h[i] > 0.003 * close[i]):          # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_H4_L4_Breakout_Volume_ADX_Filter"
timeframe = "12h"
leverage = 1.0