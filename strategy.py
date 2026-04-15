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
    
    # Calculate weekly Donchian channels (20-period)
    # Upper = highest high over past 20 weeks, Lower = lowest low over past 20 weeks
    weekly_upper = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    weekly_lower = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Align HTF Donchian levels to daily timeframe
    upper_1d = align_htf_to_ltf(prices, df_1w, weekly_upper)
    lower_1d = align_htf_to_ltf(prices, df_1w, weekly_lower)
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate daily volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Calculate daily ADX(14) for regime filter
    plus_dm = np.where((high - np.concatenate([[high[0]], high[:-1]])) > (np.concatenate([[low[0]], low[:-1]]) - low), 
                       np.maximum(high - np.concatenate([[high[0]], high[:-1]]), 0), 0)
    minus_dm = np.where((np.concatenate([[low[0]], low[:-1]]) - low) > (high - np.concatenate([[high[0]], high[:-1]])), 
                        np.maximum(np.concatenate([[low[0]], low[:-1]]) - low, 0), 0)
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (tr_14 + 1e-10)
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (tr_14 + 1e-10)
    dx_14 = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_14 = pd.Series(dx_14).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_1d[i]) or np.isnan(lower_1d[i]) or 
            np.isnan(atr_14[i]) or np.isnan(volume_ratio[i]) or np.isnan(adx_14[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Daily price breaks above weekly Donchian upper with volume confirmation → long
        # 2. Daily price breaks below weekly Donchian lower with volume confirmation → short
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Regime filter: ADX > 25 (trending market)
        # 5. Volatility filter: ATR > 0.3% of price (avoid extremely low volatility)
        # 6. Discrete position sizing: 0.25
        
        # Long conditions: daily breakout above weekly Donchian upper
        if (close[i] > upper_1d[i] and            # Daily price above weekly Donchian upper
            volume_ratio[i] > 1.5 and             # Volume confirmation
            adx_14[i] > 25 and                    # Trending regime filter
            atr_14[i] > 0.003 * close[i]):        # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: daily breakdown below weekly Donchian lower
        elif (close[i] < lower_1d[i] and          # Daily price below weekly Donchian lower
              volume_ratio[i] > 1.5 and           # Volume confirmation
              adx_14[i] > 25 and                  # Trending regime filter
              atr_14[i] > 0.003 * close[i]):      # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyDonchian20_Breakout_Volume_ADX_Filter"
timeframe = "1d"
leverage = 1.0