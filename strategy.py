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
    
    # Calculate daily Camarilla pivot levels (based on previous day)
    # Camarilla: R4 = Close + ((High-Low) * 1.1/2), R3 = Close + ((High-Low) * 1.1/4)
    #          S3 = Close - ((High-Low) * 1.1/4), S4 = Close - ((High-Low) * 1.1/2)
    daily_range = daily_high - daily_low
    r4 = daily_close + (daily_range * 1.1 / 2)
    r3 = daily_close + (daily_range * 1.1 / 4)
    s3 = daily_close - (daily_range * 1.1 / 4)
    s4 = daily_close - (daily_range * 1.1 / 2)
    
    # Align HTF Camarilla levels to 4h timeframe
    r4_4h = align_htf_to_ltf(prices, df_1d, r4)
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    s4_4h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 4h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Calculate 4h ADX(14) for trend strength filter
    plus_dm = np.where((high - np.concatenate([[high[0]], high[:-1]])) > (np.concatenate([[low[0]], low[:-1]]) - low), 
                       np.maximum(high - np.concatenate([[high[0]], high[:-1]]), 0), 0)
    minus_dm = np.where((np.concatenate([[low[0]], low[:-1]]) - low) > (high - np.concatenate([[high[0]], high[:-1]])), 
                        np.maximum(np.concatenate([[low[0]], low[:-1]]) - low, 0), 0)
    tr_smooth = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean() / tr_smooth
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean() / tr_smooth
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_4h[i]) or np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or 
            np.isnan(s4_4h[i]) or np.isnan(atr_14[i]) or np.isnan(volume_ratio[i]) or 
            np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 4h price breaks above R4 with volume confirmation AND strong trend (ADX > 25) → long
        # 2. 4h price breaks below S4 with volume confirmation AND strong trend (ADX > 25) → short
        # 3. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        # 4. Volume confirmation: volume > 1.3x average
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: 4h breakout above R4 (strong continuation)
        if (close[i] > r4_4h[i] and            # 4h price above R4 Camarilla
            volume_ratio[i] > 1.3 and          # Volume confirmation
            atr_14[i] > 0.005 * close[i] and   # Volatility filter
            adx[i] > 25):                      # Strong trend filter
            signals[i] = 0.25
            
        # Short conditions: 4h breakdown below S4 (strong continuation)
        elif (close[i] < s4_4h[i] and          # 4h price below S4 Camarilla
              volume_ratio[i] > 1.3 and        # Volume confirmation
              atr_14[i] > 0.005 * close[i] and # Volatility filter
              adx[i] > 25):                    # Strong trend filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R4_S4_Breakout_Volume_ADX_Filter"
timeframe = "4h"
leverage = 1.0