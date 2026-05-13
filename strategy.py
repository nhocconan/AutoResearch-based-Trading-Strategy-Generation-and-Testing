#!/usr/bin/env python3
"""
6h_OrderBlock_Trend_Filter
Hypothesis: Order blocks formed during 12h consolidation act as support/resistance on 6h timeframe. 
In bull markets, price respects bullish order blocks; in bear markets, respects bearish order blocks.
Trades only in direction of 12h trend (EMA50) to avoid counter-trend whipsaws. 
Volume confirmation ensures institutional participation. 
Target: 15-35 trades/year per symbol.
"""

name = "6h_OrderBlock_Trend_Filter"
timeframe = "6h"
leverage = 1.0

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
    
    # Order block detection: look for strong reversal candles after consolidation
    # Bullish OB: bearish candle followed by strong bullish candle breaking its high
    # Bearish OB: bullish candle followed by strong bearish candle breaking its low
    body_size = np.abs(close - open_)
    candle_range = high - low
    strong_candle = body_size > 0.6 * candle_range  # strong close relative to range
    
    # Bullish order block: previous candle bearish, current candle bullish and strong
    bullish_ob = (close[:-1] < open_[1:]) & (close[1:] > open_[0:-1]) & strong_candle[1:]
    bullish_ob = np.concatenate([[False], bullish_ob])  # align with current candle
    
    # Bearish order block: previous candle bullish, current candle bearish and strong
    bearish_ob = (close[:-1] > open_[1:]) & (close[1:] < open_[0:-1]) & strong_candle[1:]
    bearish_ob = np.concatenate([[False], bearish_ob])
    
    # Store OB levels: for bullish OB, use the low of the bearish candle; for bearish OB, use the high of the bullish candle
    bullish_ob_level = np.where(bullish_ob, low[:-1], np.nan)  # low of prior bearish candle
    bearish_ob_level = np.where(bearish_ob, high[:-1], np.nan)  # high of prior bullish candle
    
    # Forward fill OB levels until they are broken
    bullish_ob_level = pd.Series(bullish_ob_level).ffill().values
    bearish_ob_level = pd.Series(bearish_ob_level).ffill().values
    
    # 12h trend filter: EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_12h = df_12h['close'].values > ema_50_12h
    downtrend_12h = df_12h['close'].values < ema_50_12h
    uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h)
    downtrend_12h_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h)
    
    # Volume confirmation: volume > 1.8 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.8 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        bull_ob = bullish_ob_level[i]
        bear_ob = bearish_ob_level[i]
        uptrend = uptrend_12h_aligned[i]
        downtrend = downtrend_12h_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: price holds above bullish OB, 12h uptrend, volume confirmation
            if not np.isnan(bull_ob) and close[i] > bull_ob and uptrend and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: price holds below bearish OB, 12h downtrend, volume confirmation
            elif not np.isnan(bear_ob) and close[i] < bear_ob and downtrend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below bullish OB or 12h trend turns down
            if close[i] < bull_ob or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above bearish OB or 12h trend turns up
            if close[i] > bear_ob or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals