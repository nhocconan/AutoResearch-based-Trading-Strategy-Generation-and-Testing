#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_Trend_Momentum_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h data ONCE before loop for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 4h EMA50 for trend direction ===
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1d EMA200 for long-term trend filter ===
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === 1h RSI for momentum ===
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    loss_ma = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = gain_ma / np.where(loss_ma > 0, loss_ma, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1h Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    # Precompute session filter (8-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if outside session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = prices['close'].iloc[i]
        ema_4h_val = ema_50_4h_aligned[i]
        ema_1d_val = ema_200_1d_aligned[i]
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_4h_val) or np.isnan(ema_1d_val) or 
            np.isnan(rsi_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above 4h EMA50 (uptrend) + price above 1d EMA200 (long-term uptrend) 
            #         + RSI > 55 (bullish momentum) + volume spike
            if (close_val > ema_4h_val and close_val > ema_1d_val and 
                rsi_val > 55 and vol_ratio_val > 1.8):
                signals[i] = 0.20
                position = 1
                entry_price = close_val
            # Short: price below 4h EMA50 (downtrend) + price below 1d EMA200 (long-term downtrend) 
            #          + RSI < 45 (bearish momentum) + volume spike
            elif (close_val < ema_4h_val and close_val < ema_1d_val and 
                  rsi_val < 45 and vol_ratio_val > 1.8):
                signals[i] = -0.20
                position = -1
                entry_price = close_val
        
        elif position == 1:
            # Long exit: trend reversal (4h or 1d) or momentum fade
            if (close_val <= ema_4h_val or close_val <= ema_1d_val or rsi_val < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: trend reversal (4h or 1d) or momentum fade
            if (close_val >= ema_4h_val or close_val >= ema_1d_val or rsi_val > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals