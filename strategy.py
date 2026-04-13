#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h primary timeframe with 4h/1d HTF filters
    # Long: 4h EMA21 uptrend + 1d close > SMA50 + 1h bullish engulfing candle + session 08-20 UTC
    # Short: 4h EMA21 downtrend + 1d close < SMA50 + 1h bearish engulfing candle + session 08-20 UTC
    # Exit: opposite engulfing candle or 4h EMA cross reversal
    # Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag
    # Uses HTF for direction, 1h for precise timing, session filter to reduce noise
    # Discrete position sizing: 0.20 to minimize fee churn
    
    close = prices['close'].values
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for trend filter (EMA21)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_4h_21 = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_4h_21_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_21)
    
    # Get 1d data for trend filter (SMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    sma_1d_50 = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_50)
    
    # Calculate 1h engulfing candles
    bullish_engulfing = (close > open_price) & (open_price <= np.roll(close, 1)) & (close >= np.roll(open_price, 1)) & ((close - open_price) > (np.roll(close, 1) - np.roll(open_price, 1)))
    bearish_engulfing = (close < open_price) & (open_price >= np.roll(close, 1)) & (close <= np.roll(open_price, 1)) & ((open_price - close) > (np.roll(open_price, 1) - np.roll(close, 1)))
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% position size
    
    for i in range(50, n):  # start from 50 to have enough data for calculations
        # Skip if data not ready
        if (np.isnan(ema_4h_21_aligned[i]) or 
            np.isnan(sma_1d_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # HTF trend conditions
        ema_uptrend = ema_4h_21_aligned[i] > ema_4h_21_aligned[i-1] if i > 0 else False
        ema_downtrend = ema_4h_21_aligned[i] < ema_4h_21_aligned[i-1] if i > 0 else False
        price_above_sma = close_1d[i] > sma_1d_50_aligned[i]
        price_below_sma = close_1d[i] < sma_1d_50_aligned[i]
        
        # Entry conditions
        enter_long = ema_uptrend and price_above_sma and bullish_engulfing[i] and in_session[i]
        enter_short = ema_downtrend and price_below_sma and bearish_engulfing[i] and in_session[i]
        
        # Exit conditions: opposite engulfing or HTF trend reversal
        exit_long = position == 1 and (bearish_engulfing[i] or not ema_uptrend)
        exit_short = position == -1 and (bullish_engulfing[i] or not ema_downtrend)
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_ema_sma_engulfing_session_v1"
timeframe = "1h"
leverage = 1.0