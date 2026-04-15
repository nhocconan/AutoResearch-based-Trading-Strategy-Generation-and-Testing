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
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr3 = np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr_daily = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_daily_14 = pd.Series(tr_daily).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align daily ATR to 12h timeframe
    atr_daily_14_12h = align_htf_to_ltf(prices, df_1d, atr_daily_14)
    
    # Calculate 12h EMA(34) for trend filter
    ema_34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_daily_14_12h[i]) or np.isnan(ema_34[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 12h price breaks above daily ATR-based upper band with volume confirmation → long
        # 2. 12h price breaks below daily ATR-based lower band with volume confirmation → short
        # 3. Trend filter: price > EMA34 for long, price < EMA34 for short
        # 4. Volume confirmation: volume > 1.5x average
        # 5. Volatility filter: daily ATR > 1% of price (avoid low volatility chop)
        # 6. Discrete position sizing: 0.25
        
        # Dynamic bands based on daily ATR
        upper_band = close[i-1] + (atr_daily_14_12h[i] * 1.5)
        lower_band = close[i-1] - (atr_daily_14_12h[i] * 1.5)
        
        # Long conditions: 12h breakout above upper band with volume and trend confirmation
        if (close[i] > upper_band and            # 12h price above upper ATR band
            close[i] > ema_34[i] and             # Trend filter: above EMA34
            volume_ratio[i] > 1.5 and            # Volume confirmation
            atr_daily_14_12h[i] > 0.01 * close[i]):  # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: 12h breakdown below lower band with volume and trend confirmation
        elif (close[i] < lower_band and          # 12h price below lower ATR band
              close[i] < ema_34[i] and           # Trend filter: below EMA34
              volume_ratio[i] > 1.5 and          # Volume confirmation
              atr_daily_14_12h[i] > 0.01 * close[i]):  # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_ATR_Breakout_EMA34_Volume_Filter"
timeframe = "12h"
leverage = 1.0