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
    
    # Calculate daily Donchian(20) channels (based on previous 20 days)
    donchian_upper = pd.Series(daily_high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = pd.Series(daily_low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align HTF Donchian levels to 4h timeframe
    donchian_upper_4h = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_4h = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Calculate 4h ATR(14) for volatility filter and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_4h[i]) or np.isnan(donchian_lower_4h[i]) or 
            np.isnan(atr_14[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 4h price breaks above daily Donchian upper with volume confirmation → long
        # 2. 4h price breaks below daily Donchian lower with volume confirmation → short
        # 3. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        # 4. Volume confirmation: volume > 1.3x average
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: 4h breakout above daily Donchian upper
        if (close[i] > donchian_upper_4h[i] and            # 4h price above daily Donchian upper
            volume_ratio[i] > 1.3 and                      # Volume confirmation
            atr_14[i] > 0.005 * close[i]):                 # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: 4h breakdown below daily Donchian lower
        elif (close[i] < donchian_lower_4h[i] and          # 4h price below daily Donchian lower
              volume_ratio[i] > 1.3 and                    # Volume confirmation
              atr_14[i] > 0.005 * close[i]):               # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_DailyDonchian20_Breakout_Volume_ATR_Filter"
timeframe = "4h"
leverage = 1.0