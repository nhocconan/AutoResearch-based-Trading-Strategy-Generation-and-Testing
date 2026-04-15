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
    # Upper band = highest high over 20 weeks
    # Lower band = lowest low over 20 weeks
    weekly_upper = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    weekly_lower = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Align HTF Donchian levels to daily timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1w, weekly_upper)
    lower_aligned = align_htf_to_ltf(prices, df_1w, weekly_lower)
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate daily volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(atr_14[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Daily price breaks above weekly Donchian upper with volume confirmation → long
        # 2. Daily price breaks below weekly Donchian lower with volume confirmation → short
        # 3. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        # 4. Volume confirmation: volume > 1.3x average
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: daily breakout above weekly upper band
        if (close[i] > upper_aligned[i] and            # daily price above weekly upper
            volume_ratio[i] > 1.3 and                  # volume confirmation
            atr_14[i] > 0.005 * close[i]):             # volatility filter
            signals[i] = 0.25
            
        # Short conditions: daily breakdown below weekly lower band
        elif (close[i] < lower_aligned[i] and          # daily price below weekly lower
              volume_ratio[i] > 1.3 and                # volume confirmation
              atr_14[i] > 0.005 * close[i]):           # volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyDonchian20_Breakout_Volume_ATR_Filter_v2"
timeframe = "1d"
leverage = 1.0