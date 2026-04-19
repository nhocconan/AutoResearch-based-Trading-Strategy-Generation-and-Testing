#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_RSI_Pullback_Strategy"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 20-period EMA for 4h trend
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Get 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 14-period RSI for 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align 1d RSI to 1h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume confirmation: current volume > 1.8x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30
    
    for i in range(start_idx, n):
        if np.isnan(ema20_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma_24[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_24[i]
        ema_trend = ema20_4h_aligned[i]
        rsi = rsi_1d_aligned[i]
        
        volume_confirmed = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long: Pullback in uptrend - RSI oversold + price above 4h EMA20 + volume
            if rsi < 30 and price > ema_trend and volume_confirmed:
                signals[i] = 0.20
                position = 1
            # Short: Pullback in downtrend - RSI overbought + price below 4h EMA20 + volume
            elif rsi > 70 and price < ema_trend and volume_confirmed:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: RSI overbought or price breaks below 4h EMA20
            if rsi > 70 or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: RSI oversold or price breaks above 4h EMA20
            if rsi < 30 or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals