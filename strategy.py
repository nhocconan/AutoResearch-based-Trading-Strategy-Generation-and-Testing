#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_RSI_Divergence_Trend_Confirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 14-period RSI on daily
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align daily RSI to 4h timeframe
    rsi_4h = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 4h EMA34 for trend filter
    ema_34 = pd.Series(close).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(rsi_4h[i]) or np.isnan(ema_34[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        rsi = rsi_4h[i]
        ema = ema_34[i]
        
        volume_confirmed = vol > 1.8 * vol_ma
        
        if position == 0:
            # Bullish divergence setup: RSI < 30 (oversold) + price above EMA34 (uptrend bias)
            if rsi < 30 and price > ema and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Bearish divergence setup: RSI > 70 (overbought) + price below EMA34 (downtrend bias)
            elif rsi > 70 and price < ema and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: RSI returns to neutral (50) or price breaks below EMA34
            if rsi >= 50 or price < ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: RSI returns to neutral (50) or price breaks above EMA34
            if rsi <= 50 or price > ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals