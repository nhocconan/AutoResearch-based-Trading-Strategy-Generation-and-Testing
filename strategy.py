#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_RVOL_Pullback_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_6h = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 6h EMA34 for pullback entries
    ema34_6h = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 6h RSI(14) for momentum
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # 6h Volume: current > 2.0 x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 150
    
    for i in range(start_idx, n):
        if np.isnan(ema200_6h[i]) or np.isnan(ema34_6h[i]) or np.isnan(rsi[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema34 = ema34_6h[i]
        ema200 = ema200_6h[i]
        rsi_val = rsi[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        volume_confirmed = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long: Above EMA200, pullback to EMA34 with RSI<40 and volume
            if price > ema200 and price <= ema34 and rsi_val < 40 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Below EMA200, pullback to EMA34 with RSI>60 and volume
            elif price < ema200 and price >= ema34 and rsi_val > 60 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price crosses below EMA34 or RSI>60
            if price < ema34 or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price crosses above EMA34 or RSI<40
            if price > ema34 or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals