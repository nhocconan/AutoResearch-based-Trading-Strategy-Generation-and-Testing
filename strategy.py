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
    
    # Calculate daily EMA(34) for trend filter
    daily_close = df_1d['close'].values
    ema_34_daily = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF EMA to 6h timeframe
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34_daily)
    
    # Get weekly HTF data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate weekly pivot points (standard: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align HTF weekly pivots to 6h timeframe
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_6h = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_6h = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Calculate 6h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_6h[i]) or np.isnan(weekly_pivot_6h[i]) or 
            np.isnan(weekly_r1_6h[i]) or np.isnan(weekly_s1_6h[i]) or 
            np.isnan(atr_14[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # Long: Price > daily EMA34 (uptrend) AND breaks above weekly R1 with volume confirmation
        # Short: Price < daily EMA34 (downtrend) AND breaks below weekly S1 with volume confirmation
        # Volatility filter: ATR > 0.5% of price
        # Volume confirmation: volume > 1.3x average
        # Discrete position sizing: 0.25
        
        # Long conditions
        if (close[i] > ema_34_6h[i] and           # Price above daily EMA34 (uptrend)
            close[i] > weekly_r1_6h[i] and        # 6h price breaks above weekly R1
            volume_ratio[i] > 1.3 and             # Volume confirmation
            atr_14[i] > 0.005 * close[i]):        # Volatility filter
            signals[i] = 0.25
            
        # Short conditions
        elif (close[i] < ema_34_6h[i] and         # Price below daily EMA34 (downtrend)
              close[i] < weekly_s1_6h[i] and      # 6h price breaks below weekly S1
              volume_ratio[i] > 1.3 and           # Volume confirmation
              atr_14[i] > 0.005 * close[i]):      # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_EMA34_WeeklyPivot_R1S1_Breakout_Volume_ATR_Filter"
timeframe = "6h"
leverage = 1.0