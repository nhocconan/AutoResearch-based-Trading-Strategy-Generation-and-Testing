#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Keltner_RSI_Trend_4h"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1h data for trend filter (more responsive than 12h)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 50:
        return np.zeros(n)
    
    # Get 1d data for Keltner calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous day's data for Keltner calculation
    prev_high = df_1h['high'].shift(1).values
    prev_low = df_1h['low'].shift(1).values
    prev_close = df_1h['close'].shift(1).values
    
    # Calculate ATR for Keltner (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[np.nan], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[np.nan], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Keltner Channels (14-period ATR, multiplier 2.0)
    keltner_upper = prev_close + (atr * 2.0)
    keltner_lower = prev_close - (atr * 2.0)
    
    # Trend filter: 1h EMA50
    ema50_1h = pd.Series(df_1h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # RSI (14-period)
    delta = np.diff(close, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, np.nan, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # Align all to 4h (primary timeframe)
    keltner_upper_4h = align_htf_to_ltf(prices, df_1h, keltner_upper)
    keltner_lower_4h = align_htf_to_ltf(prices, df_1h, keltner_lower)
    ema50_1h_4h = align_htf_to_ltf(prices, df_1h, ema50_1h)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 14)  # Need enough data for EMA50 and RSI
    
    for i in range(start_idx, n):
        if (np.isnan(keltner_upper_4h[i]) or np.isnan(keltner_lower_4h[i]) or
            np.isnan(ema50_1h_4h[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper = keltner_upper_4h[i]
        lower = keltner_lower_4h[i]
        trend = ema50_1h_4h[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Enter long: price above upper Keltner with bullish trend and RSI > 50
            if close[i] > upper and close[i] > trend and rsi_val > 50:
                signals[i] = 0.25
                position = 1
            # Enter short: price below lower Keltner with bearish trend and RSI < 50
            elif close[i] < lower and close[i] < trend and rsi_val < 50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below lower Keltner or RSI < 40
            if close[i] < lower or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above upper Keltner or RSI > 60
            if close[i] > upper or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals