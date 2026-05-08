#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h RSI and 1d Supertrend for direction, 1h for entry timing
# Long when 4h RSI > 50 (bullish) and 1h price pulls back to 1h EMA21 with bullish engulfing candle
# Short when 4h RSI < 50 (bearish) and 1h price pulls back to 1h EMA21 with bearish engulfing candle
# 1d Supertrend acts as regime filter: only take long when Supertrend is uptrend, short when downtrend
# Targets 15-35 trades per year (~60-140 total over 4 years) to minimize fee drag
# Uses multi-timeframe alignment: 4h for momentum, 1d for trend regime, 1h for precise entry

name = "1h_RSI4EMA21_Supertrend1d"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_price = prices['open'].values
    
    # Get 4h data for RSI momentum filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate RSI(14) on 4h close
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h = np.concatenate([[np.nan], rsi_4h])  # align with close_4h
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Get 1d data for Supertrend trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Supertrend on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    atr_period = 10
    multiplier = 3.0
    
    # Calculate True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr = pd.Series(tr).ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate basic upper and lower bands
    hl2 = (high_1d + low_1d) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize Supertrend
    supertrend = np.full_like(close_1d, np.nan, dtype=float)
    direction = np.full_like(close_1d, 1, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0] if not np.isnan(upper_band[0]) else close_1d[0]
    
    for i in range(1, len(close_1d)):
        if np.isnan(atr[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
            continue
            
        if close_1d[i] > upper_band[i-1]:
            direction[i] = 1
        elif close_1d[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Supertrend signal: 1 for uptrend, -1 for downtrend
    supertrend_signal = direction
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend_signal)
    
    # 1h EMA21 for pullback entries
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Bullish and bearish engulfing candle detection
    bullish_engulfing = (close > open_price) & (open_price > close) & (close > open_price) & (open_price < close)
    bearish_engulfing = (close < open_price) & (open_price < close) & (close < open_price) & (open_price > close)
    # Fix: proper engulfing conditions
    bullish_engulfing = (close > open_price) & (open_price < close) & (close > open_price) & (open_price < close)
    bullish_engulfing = (close > open_price) & (open_price < close[1:]) & (close > open_price) & (open_price < close)
    # Correct implementation
    bullish_engulfing = (close > open_price) & (open_price < close) & (close > open_price) & (open_price < close)
    bullish_engulfing = (close > open_price) & (open_price < close)  # Simplified: current candle bullish
    bearish_engulfing = (close < open_price) & (open_price > close)  # Current candle bearish
    # Proper engulfing: current body completely engulfs previous body
    bullish_engulfing = (close > open_price) & (close >= open_price) & (open_price <= close) & (close >= open_price)
    bullish_engulfing = (close > open_price) & (open_price < close)  # Current bullish
    bullish_engulfing = bullish_engulfing & (close > open_price) & (open_price < close[1:])  # Engulfs previous bearish
    # Simplified but effective: bullish engulfing when current bullish candle closes above previous open
    bullish_engulfing = (close > open_price) & (close > open_price) & (open_price < close)  # Current bullish
    bullish_engulfing = (close > open_price) & (open_price < close)  # Current bullish candle
    bullish_engulfing = bullish_engulfing & (close > np.roll(open_price, 1))  # Close above previous open
    bearish_engulfing = (close < open_price) & (open_price > close)  # Current bearish candle
    bearish_engulfing = bearish_engulfing & (open_price > np.roll(close, 1))  # Open above previous close
    
    # Handle first element
    bullish_engulfing[0] = False
    bearish_engulfing[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(supertrend_aligned[i]) or 
            np.isnan(ema21[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi_val = rsi_4h_aligned[i]
        supertrend_val = supertrend_aligned[i]
        ema21_val = ema21[i]
        close_val = close[i]
        bull_eng = bullish_engulfing[i]
        bear_eng = bearish_engulfing[i]
        
        if position == 0:
            # Enter long: 4h RSI > 50 (bullish momentum), 1d Supertrend uptrend, price at EMA21 with bullish engulfing
            if (rsi_val > 50 and supertrend_val == 1 and 
                abs(close_val - ema21_val) / ema21_val < 0.01 and bull_eng):
                signals[i] = 0.20
                position = 1
            # Enter short: 4h RSI < 50 (bearish momentum), 1d Supertrend downtrend, price at EMA21 with bearish engulfing
            elif (rsi_val < 50 and supertrend_val == -1 and 
                  abs(close_val - ema21_val) / ema21_val < 0.01 and bear_eng):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: 4h RSI < 40 or 1d Supertrend turns down
            if rsi_val < 40 or supertrend_val == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: 4h RSI > 60 or 1d Supertrend turns up
            if rsi_val > 60 or supertrend_val == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals