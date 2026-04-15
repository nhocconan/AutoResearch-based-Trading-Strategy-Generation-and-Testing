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
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly ATR(14) for volatility regime filter
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = np.abs(df_1w['high'] - np.concatenate([[df_1w['close'].iloc[0]], df_1w['close'].iloc[:-1]]))
    tr3 = np.abs(df_1w['low'] - np.concatenate([[df_1w['close'].iloc[0]], df_1w['close'].iloc[:-1]]))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1w = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Calculate weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate weekly RSI(14) for momentum filter
    delta = pd.Series(df_1w['close'].values).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1w = 100 - (100 / (1 + rs))
    rsi_14_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_14_1w)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_1w_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(rsi_14_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when weekly ATR is elevated (> 0.8% of price)
        vol_filter = atr_14_1w_aligned[i] > 0.008 * close[i]
        
        # Long conditions:
        # 1. Price above weekly EMA34 (bullish bias)
        # 2. Weekly RSI between 30 and 50 (oversold to neutral momentum)
        # 3. Volatility filter
        if (close[i] > ema_34_1w_aligned[i] and
            30 <= rsi_14_1w_aligned[i] <= 50 and
            vol_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below weekly EMA34 (bearish bias)
        # 2. Weekly RSI between 50 and 70 (neutral to overbought momentum)
        # 3. Volatility filter
        elif (close[i] < ema_34_1w_aligned[i] and
              50 <= rsi_14_1w_aligned[i] <= 70 and
              vol_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_EMA34_RSI14_VolFilter_1w_v1"
timeframe = "1d"
leverage = 1.0