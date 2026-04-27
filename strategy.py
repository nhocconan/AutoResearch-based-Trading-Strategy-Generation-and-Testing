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
    
    # Get 12h data for trend and 1d data for volatility filter
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 1 or len(df_1d) < 1:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 1d ATR(14) for volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - np.roll(close_1d, 1)[1:])
    tr3 = np.abs(low_1d[1:] - np.roll(close_1d, 1)[1:])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # 4h Bollinger Bands (20, 2)
    close_s = pd.Series(close)
    sma20 = close_s.rolling(window=20, min_periods=20).mean().values
    std20 = close_s.rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    
    # 4h RSI(14)
    delta = np.diff(close, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(sma20[i]) or np.isnan(std20[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Conditions
        uptrend = close[i] > ema50_12h_aligned[i]
        downtrend = close[i] < ema50_12h_aligned[i]
        low_volatility = atr14_1d_aligned[i] < np.mean(atr14_1d_aligned[max(0, i-50):i+1]) * 0.8
        near_lower_bb = close[i] <= lower_bb[i] * 1.01  # within 1% of lower band
        near_upper_bb = close[i] >= upper_bb[i] * 0.99   # within 1% of upper band
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Long: oversold + near lower BB + low volatility + uptrend
        long_signal = (rsi_oversold and near_lower_bb and low_volatility and uptrend)
        # Short: overbought + near upper BB + low volatility + downtrend
        short_signal = (rsi_overbought and near_upper_bb and low_volatility and downtrend)
        
        if long_signal and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_signal and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit: opposite RSI extreme or trend reversal
        elif position == 1 and (rsi[i] > 70 or close[i] < ema50_12h_aligned[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (rsi[i] < 30 or close[i] > ema50_12h_aligned[i]):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_RSI_BB_EMA50_12hTrend_ATR14_1dVolFilter"
timeframe = "4h"
leverage = 1.0