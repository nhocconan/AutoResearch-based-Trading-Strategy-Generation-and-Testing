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
    
    # Load 1d data (primary) and 1w data (HTF)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 5:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1w EMA(20) for HTF trend
    close_1w_series = pd.Series(close_1w)
    ema_20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1d ATR(14) for volatility filter
    high_low = high_1d - low_1d
    high_close = np.abs(np.concatenate([[np.nan], high_1d[:-1]]) - close_1d)
    low_close = np.abs(np.concatenate([[np.nan], low_1d[:-1]]) - close_1d)
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr = pd.Series(tr)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d RSI(14) for momentum
    delta = np.diff(close_1d, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    avg_gain = gain_series.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = loss_series.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align all indicators to 1d timeframe (since we're using 1d timeframe)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w, additional_delay_bars=0)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):  # Start after enough data for indicators
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(rsi_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely low volatility
        if atr_14_aligned[i] < 0.01 * close[i]:  # Less than 1% of price
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price above EMA50 (1d trend), above EMA20 (1w trend), RSI > 50
            if (close[i] > ema_50_1d_aligned[i] and 
                close[i] > ema_20_1w_aligned[i] and
                rsi_aligned[i] > 50):
                position = 1
                signals[i] = position_size
            # Short entry: price below EMA50 (1d trend), below EMA20 (1w trend), RSI < 50
            elif (close[i] < ema_50_1d_aligned[i] and 
                  close[i] < ema_20_1w_aligned[i] and
                  rsi_aligned[i] < 50):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below EMA50 or RSI < 40
            if (close[i] < ema_50_1d_aligned[i] or
                rsi_aligned[i] < 40):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above EMA50 or RSI > 60
            if (close[i] > ema_50_1d_aligned[i] or
                rsi_aligned[i] > 60):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_EMA50_EMA20_RSI_Filter_v1"
timeframe = "1d"
leverage = 1.0