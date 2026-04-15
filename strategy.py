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
    
    # Calculate daily EMA(20) for short-term trend
    ema_20_1d = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate daily EMA(50) for medium-term trend
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily RSI(14) for momentum/overbought-oversold
    delta = pd.Series(df_1d['close'].values).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d.values)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(rsi_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when daily ATR is in normal range (avoid extreme volatility)
        vol_filter = (atr_14_1d_aligned[i] > 0.008 * close[i]) & (atr_14_1d_aligned[i] < 0.03 * close[i])
        
        # Trend alignment: EMA20 above EMA50 for bullish bias, below for bearish bias
        bullish_trend = ema_20_1d_aligned[i] > ema_50_1d_aligned[i]
        bearish_trend = ema_20_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Momentum filter: RSI not extreme to avoid buying tops/selling bottoms
        momentum_filter = (rsi_14_1d_aligned[i] > 30) & (rsi_14_1d_aligned[i] < 70)
        
        # Long conditions:
        # 1. Bullish trend alignment
        # 2. Price above EMA20 (confirming short-term strength)
        # 3. Normal volatility
        # 4. Healthy momentum
        if (bullish_trend and
            close[i] > ema_20_1d_aligned[i] and
            vol_filter and
            momentum_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Bearish trend alignment
        # 2. Price below EMA20 (confirming short-term weakness)
        # 3. Normal volatility
        # 4. Healthy momentum
        elif (bearish_trend and
              close[i] < ema_20_1d_aligned[i] and
              vol_filter and
              momentum_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_EMA20_EMA50_RSI_VolFilter_v1"
timeframe = "6h"
leverage = 1.0