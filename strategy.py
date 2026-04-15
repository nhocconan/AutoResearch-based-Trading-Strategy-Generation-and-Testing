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
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(34) for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_6h = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d HTF data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from prior day
    # Camarilla: based on prior day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day values (shifted by 1 to avoid look-ahead)
    prior_high = np.concatenate([[np.nan], high_1d[:-1]])
    prior_low = np.concatenate([[np.nan], low_1d[:-1]])
    prior_close = np.concatenate([[np.nan], close_1d[:-1]])
    
    # Camarilla calculations
    camarilla_pivot = (prior_high + prior_low + prior_close) / 3.0
    camarilla_range = prior_high - prior_low
    camarilla_r3 = camarilla_pivot + (camarilla_range * 1.1 / 4.0)  # R3
    camarilla_s3 = camarilla_pivot - (camarilla_range * 1.1 / 4.0)  # S3
    camarilla_r4 = camarilla_pivot + (camarilla_range * 1.1 / 2.0)  # R4
    camarilla_s4 = camarilla_pivot - (camarilla_range * 1.1 / 2.0)  # S4
    
    # Align Camarilla levels to 6h
    camarilla_pivot_6h = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_r3_6h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_6h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_6h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_6h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate 6h ATR(20) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_20 = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Session filter: trade during active UTC hours (00-24 covers all, but structure)
    hours = prices.index.hour
    in_session = (hours >= 0) & (hours <= 23)  # Always true for 6h, kept for structure
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_6h[i]) or np.isnan(camarilla_pivot_6h[i]) or 
            np.isnan(camarilla_r3_6h[i]) or np.isnan(camarilla_s3_6h[i]) or 
            np.isnan(camarilla_r4_6h[i]) or np.isnan(camarilla_s4_6h[i]) or 
            np.isnan(atr_20[i]) or np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Price above weekly EMA(34) (bullish weekly trend)
        # 2. Price breaks above Camarilla R3 with volume (breakout continuation)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.3% of price (avoid extremely low volatility)
        if (close[i] > ema_34_6h[i] and
            close[i] > camarilla_r3_6h[i] and
            volume_ratio[i] > 1.5 and
            atr_20[i] > 0.003 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below weekly EMA(34) (bearish weekly trend)
        # 2. Price breaks below Camarilla S3 with volume (breakdown continuation)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.3% of price
        elif (close[i] < ema_34_6h[i] and
              close[i] < camarilla_s3_6h[i] and
              volume_ratio[i] > 1.5 and
              atr_20[i] > 0.003 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_1w_EMA34_1d_Camarilla_R3S3_Breakout_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0