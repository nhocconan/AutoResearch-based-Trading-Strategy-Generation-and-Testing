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
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA(34) for trend
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 12h ATR(14) for volatility filter
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = np.abs(df_12h['high'] - np.concatenate([[df_12h['close'].iloc[0]], df_12h['close'].iloc[:-1]]))
    tr3 = np.abs(df_12h['low'] - np.concatenate([[df_12h['close'].iloc[0]], df_12h['close'].iloc[:-1]]))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    
    # Calculate 12h RSI(14) for momentum
    delta = pd.Series(df_12h['close'].values).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_14_12h = 100 - (100 / (1 + rs))
    rsi_14_12h = rsi_14_12h.fillna(50).values
    rsi_14_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_14_12h)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(atr_14_12h_aligned[i]) or 
            np.isnan(rsi_14_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when 12h ATR is elevated (> 0.4% of price)
        vol_filter = atr_14_12h_aligned[i] > 0.004 * close[i]
        
        # Momentum filter: RSI between 30 and 70 to avoid extremes
        mom_filter = (rsi_14_12h_aligned[i] >= 30) & (rsi_14_12h_aligned[i] <= 70)
        
        # Long conditions:
        # 1. Price above 12h EMA34 (bullish bias)
        # 2. Volatility filter
        # 3. Momentum filter
        if (close[i] > ema_34_12h_aligned[i] and
            vol_filter and
            mom_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below 12h EMA34 (bearish bias)
        # 2. Volatility filter
        # 3. Momentum filter
        elif (close[i] < ema_34_12h_aligned[i] and
              vol_filter and
              mom_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_EMA34_RSI14_VolFilter_v1"
timeframe = "6h"
leverage = 1.0