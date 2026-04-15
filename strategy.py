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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily RSI(14) for momentum filter
    delta = pd.Series(df_1d['close'].values).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d = rsi_14_1d.fillna(50).values  # neutral when undefined
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Calculate daily ADX(14) for trend strength filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    plus_dm = df_1d['high'].diff()
    minus_dm = df_1d['low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr_14 = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean()
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean() / tr_14
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean() / tr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_14_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Calculate daily ATR(14) for volatility normalization
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_14_1d_aligned[i]) or np.isnan(adx_14_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when ADX > 25 (trending market)
        trend_filter = adx_14_1d_aligned[i] > 25
        
        # Volatility filter: avoid extreme volatility spikes
        vol_filter = atr_14_1d_aligned[i] < 0.08 * close[i]  # ATR < 8% of price
        
        # Long conditions:
        # 1. RSI > 50 (bullish momentum)
        # 2. Price closes above previous 12h high (breakout)
        # 3. Trend and volatility filters
        if (rsi_14_1d_aligned[i] > 50 and
            close[i] > high[i-1] and
            trend_filter and
            vol_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. RSI < 50 (bearish momentum)
        # 2. Price closes below previous 12h low (breakdown)
        # 3. Trend and volatility filters
        elif (rsi_14_1d_aligned[i] < 50 and
              close[i] < low[i-1] and
              trend_filter and
              vol_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_RSI14_ADX14_Breakout_v1"
timeframe = "12h"
leverage = 1.0