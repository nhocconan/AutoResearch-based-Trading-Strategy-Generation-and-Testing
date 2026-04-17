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
    
    # Get 1d data for ATR and RSI
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on daily
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr = np.concatenate([[np.nan], tr2])
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate RSI(14) on daily
    delta = np.diff(close_1d, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, np.nan, avg_loss)
    rsi14_1d = 100 - (100 / (1 + rs))
    
    # Align daily indicators to 1h
    atr14_1h = align_htf_to_ltf(prices, df_1d, atr14_1d)
    rsi14_1h = align_htf_to_ltf(prices, df_1d, rsi14_1d)
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1h = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1h volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hour >= 8) & (hour <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # Need sufficient data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr14_1h[i]) or 
            np.isnan(rsi14_1h[i]) or 
            np.isnan(ema50_1h[i]) or 
            np.isnan(volume_ma20[i]) or 
            np.isnan(session_filter[i])):
            signals[i] = 0.0
            continue
        
        # Skip outside trading session
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below 4h EMA50
        price_above_ema = close[i] > ema50_1h[i]
        price_below_ema = close[i] < ema50_1h[i]
        
        # Mean reversion condition: RSI extreme
        rsi_oversold = rsi14_1h[i] < 30
        rsi_overbought = rsi14_1h[i] > 70
        
        if position == 0:
            # Long: Oversold RSI + price above 4h EMA50 + volume
            if (rsi_oversold and price_above_ema and volume_filter):
                signals[i] = 0.20
                position = 1
            # Short: Overbought RSI + price below 4h EMA50 + volume
            elif (rsi_overbought and price_below_ema and volume_filter):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: RSI overbought OR price crosses below 4h EMA50
            if (rsi14_1h[i] > 70) or (close[i] < ema50_1h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI oversold OR price crosses above 4h EMA50
            if (rsi14_1h[i] < 30) or (close[i] > ema50_1h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI_MeanReversion_EMA50_Trend"
timeframe = "1h"
leverage = 1.0