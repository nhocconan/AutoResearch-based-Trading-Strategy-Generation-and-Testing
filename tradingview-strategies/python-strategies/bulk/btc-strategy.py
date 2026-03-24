#!/usr/bin/env python3
import numpy as np
import pandas as pd

name = "BTC Strategy"
timeframe = "1d"
leverage = 1

def ema(series, length):
    alpha = 2.0 / (length + 1.0)
    return series.ewm(alpha=alpha, adjust=False).mean()

def rsi(series, length):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=length).mean()
    rs = gain / loss
    return 100.0 - (100.0 / (1.0 + rs))

def stoch_rsi(series, rsi_length, stoch_length, smooth):
    rs = rsi(series, rsi_length)
    lowest_low = rs.rolling(window=stoch_length).min()
    highest_high = rs.rolling(window=stoch_length).max()
    denom = highest_high - lowest_low
    stoch = 100.0 * (rs - lowest_low) / denom
    stoch = stoch.fillna(0.0)
    return stoch.rolling(window=smooth).mean()

def sma(series, length):
    return series.rolling(window=length).mean()

def generate_signals(prices):
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("prices must be a pandas DataFrame")
    
    close = prices['close']
    low = prices['low']
    open_time = pd.to_datetime(prices['open_time'], utc=True, errors='coerce')
    
    rsi_threshold = 31
    rsi_length = 15
    srsi_length = 9
    srsi_smooth = 4
    srsi_sell_threshold = 61
    dma_length = 17
    dma_signal_threshold = 0
    macd_fast = 11
    macd_slow = 18
    macd_signal_len = 6
    macd_signal_threshold = -2
    long_loss_tol = 7.0
    
    macd_line = ema(close, macd_fast) - ema(close, macd_slow)
    macd_signal = ema(macd_line, macd_signal_len)
    delta = macd_line - macd_signal
    
    rs = rsi(close, rsi_length)
    k = stoch_rsi(close, rsi_length, srsi_length, srsi_smooth)
    
    norm = sma(close, dma_length)
    threshold = close - norm
    
    delta_cross = (delta > macd_signal_threshold) & (delta.shift(1) <= macd_signal_threshold)
    rs_cross = (rs > rsi_threshold) & (rs.shift(1) <= rsi_threshold)
    buy_cond = (delta_cross | rs_cross) & (k < srsi_sell_threshold)
    
    delta_cross_under = (delta < 0) & (delta.shift(1) >= 0)
    thresh_cross_under = (threshold < dma_signal_threshold) & (threshold.shift(1) >= dma_signal_threshold)
    sell_cond = (delta_cross_under | thresh_cross_under) & (k > srsi_sell_threshold)
    
    # Fair-comparison mode uses the shared suite window, not Pine date gates.
    buy_cond = buy_cond.fillna(False)
    sell_cond = sell_cond.fillna(False)
    
    n = len(close)
    signals = np.zeros(n, dtype=int)
    position = 0
    entry_price = 0.0
    
    for i in range(n):
        signals[i] = position
        
        if position == 1:
            stop_level = entry_price * (1.0 - long_loss_tol / 100.0)
            if low.iloc[i] < stop_level:
                position = 0
                entry_price = 0.0
                continue
            
            if sell_cond.iloc[i]:
                position = 0
                entry_price = 0.0
                continue
                
        if position == 0 and buy_cond.iloc[i]:
            position = 1
            entry_price = close.iloc[i]
            
    return signals
