#!/usr/bin/env python3
"""
6h_12h_1d_engulfing_momentum_v1
Strategy: 6h engulfing candles with 12h/1d momentum filters
Timeframe: 6h
Leverage: 1.0
Hypothesis: Bullish/bearish engulfing patterns on 6h timeframe, when aligned with 12h momentum (price > 20-period EMA) and 1d trend (price > 50-period EMA), provide high-probability entries. This captures momentum shifts while avoiding counter-trend trades. Works in bull markets by catching continuations and in bear markets by fading overextended moves into key levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_engulfing_momentum_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 6h EMA for momentum filter
    close_6h = close
    ema_20_6h = pd.Series(close_6h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_6h = pd.Series(close_6h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 6h ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 12h EMA (trend filter) ===
    close_12h = df_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # === 1d EMA (trend filter) ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Session filter: 0-23 UTC (covers major sessions)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 0) & (hours <= 23)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(ema_20_6h[i]) or np.isnan(ema_50_6h[i]) or
            np.isnan(atr_6h[i]) or np.isnan(ema_20_12h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_open = high[i] - (high[i] - low[i])  # approximate open from high-low range
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        
        # Bullish engulfing: current green candle engulfs previous red candle
        prev_close = close[i-1]
        prev_open = high[i-1] - (high[i-1] - low[i-1])  # approximate previous open
        
        bullish_engulfing = (price_close > price_open and  # current candle is green
                           prev_close < prev_open and      # previous candle was red
                           price_close >= prev_open and    # current close >= previous open
                           price_open <= prev_close)       # current open <= previous close
        
        # Bearish engulfing: current red candle engulfs previous green candle
        bearish_engulfing = (price_close < price_open and  # current candle is red
                           prev_close > prev_open and      # previous candle was green
                           price_close <= prev_open and    # current close <= previous open
                           price_open >= prev_close)       # current open >= previous close
        
        # Momentum filters
        uptrend_12h = price_close > ema_20_12h_aligned[i]
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_12h = price_close < ema_20_12h_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        
        # Long conditions: bullish engulfing + 12h uptrend + 1d uptrend
        long_signal = bullish_engulfing and uptrend_12h and uptrend_1d
        
        # Short conditions: bearish engulfing + 12h downtrend + 1d downtrend
        short_signal = bearish_engulfing and downtrend_12h and downtrend_1d
        
        # Exit conditions: opposite engulfing or loss of trend
        exit_long = (bearish_engulfing and downtrend_12h) or (price_close < ema_20_6h[i])
        exit_short = (bullish_engulfing and uptrend_12h) or (price_close > ema_50_6h[i])
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Bullish/bearish engulfing patterns on 6h timeframe, when aligned with 12h momentum (price > 20-period EMA) and 1d trend (price > 50-period EMA), provide high-probability entries. This captures momentum shifts while avoiding counter-trend trades. Works in bull markets by catching continuations and in bear markets by fading overextended moves into key levels.