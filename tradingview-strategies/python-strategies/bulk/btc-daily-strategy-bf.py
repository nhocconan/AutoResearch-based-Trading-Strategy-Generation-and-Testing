#!/usr/bin/env python3
import pandas as pd
import numpy as np

name = "BTC Daily Strategy BF"
timeframe = "1d"
leverage = 1

def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def rsi(series, length):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    res = 100 - (100 / (1 + rs))
    return res.fillna(50)

def stoch_rsi(rs, length, smooth):
    lowest = rs.rolling(window=length).min()
    highest = rs.rolling(window=length).max()
    denominator = highest - lowest
    k = 100 * (rs - lowest) / denominator.replace(0, np.nan)
    return k.rolling(window=smooth).mean().fillna(0)

def sma(series, length):
    return series.rolling(window=length).mean()

def generate_signals(prices):
    df = prices.copy()
    # Repo price data already carries timezone-aware UTC timestamps.
    # Do not preserve Pine backtest-window gating in suite mode because the
    # TradingView suite scores strategies on the shared 2021-now window.
    df['date'] = pd.to_datetime(df['open_time'], utc=True)
    
    rsi_threshold = 30
    rsi_length = 4
    srsi_length = 8
    srsi_smooth = 4
    srsi_sell_threshold = 57
    sma_length = 14
    dma_signal_threshold = -1
    fastLength = 11
    slowlength = 18
    MACDLength = 6
    MACD_signal_threshold = -2
    long_loss_tol = 5
    short_loss_tol = 5
    
    macd_line = ema(df['close'], fastLength) - ema(df['close'], slowlength)
    signal_line = ema(macd_line, MACDLength)
    delta = macd_line - signal_line
    
    rs = rsi(df['close'], rsi_length)
    k = stoch_rsi(rs, srsi_length, srsi_smooth)
    
    ohlc4 = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0
    norm = sma(ohlc4, sma_length)
    threshold = df['close'] - norm
    
    delta_cross_up = (delta > MACD_signal_threshold) & (delta.shift(1) <= MACD_signal_threshold)
    rs_cross_up = (rs > rsi_threshold) & (rs.shift(1) <= rsi_threshold)
    delta_cross_down = (delta < 0) & (delta.shift(1) >= 0)
    thresh_cross_down = (threshold < dma_signal_threshold) & (threshold.shift(1) >= dma_signal_threshold)
    
    long_cond = ((delta_cross_up | rs_cross_up) & (k < srsi_sell_threshold))
    short_cond = (delta_cross_down | (thresh_cross_down & (k > srsi_sell_threshold)))
    
    long_signals = long_cond.shift(1).fillna(False).to_numpy(dtype=bool)
    short_signals = short_cond.shift(1).fillna(False).to_numpy(dtype=bool)
    
    raw_signals = np.zeros(len(df))
    raw_signals[long_signals] = 1
    raw_signals[short_signals & ~long_signals] = -1
    
    positions = np.zeros(len(df))
    current_pos = 0
    entry_price = 0.0
    
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    
    for i in range(len(df)):
        if current_pos == 1:
            stop_level = entry_price * (1 - long_loss_tol / 100.0)
            if lows[i] <= stop_level:
                current_pos = 0
                entry_price = 0.0
        elif current_pos == -1:
            stop_level = entry_price * (1 + short_loss_tol / 100.0)
            if highs[i] >= stop_level:
                current_pos = 0
                entry_price = 0.0
        
        if current_pos == 0:
            sig = raw_signals[i]
            if sig == 1:
                current_pos = 1
                entry_price = opens[i]
            elif sig == -1:
                current_pos = -1
                entry_price = opens[i]
        
        positions[i] = current_pos
        
    return positions.astype(int)
