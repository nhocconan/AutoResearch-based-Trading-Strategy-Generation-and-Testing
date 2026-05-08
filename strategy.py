#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyEMA_Crossover_PriceAction"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA20 and EMA50 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMAs to daily timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily ATR for volatility filter and stop management
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily price action signals: engulfing candles
    bullish_engulf = (close > open_) & (open_ > close.shift(1)) & (close > close.shift(1)) & (open_ < close.shift(1))
    bearish_engulf = (close < open_) & (open_ < close.shift(1)) & (close < close.shift(1)) & (open_ > close.shift(1))
    
    # Need open prices for engulfing calculation
    open_ = prices['open'].values
    
    # Recalculate bullish/bearish engulf with proper open
    bullish_engulf = (close > open_) & (open_ < close[:-1]) & (close > open_[:-1]) & (open_ < close[:-1])
    bearish_engulf = (close < open_) & (open_ > close[:-1]) & (close < open_[:-1]) & (open_ > close[:-1])
    
    # Fix first element
    bullish_engulf[0] = False
    bearish_engulf[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish engulfing + weekly uptrend (EMA20 > EMA50)
            long_cond = bullish_engulf[i] and (ema20_1w_aligned[i] > ema50_1w_aligned[i])
            
            # Short: bearish engulfing + weekly downtrend (EMA20 < EMA50)
            short_cond = bearish_engulf[i] and (ema20_1w_aligned[i] < ema50_1w_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish engulfing OR price drops below weekly EMA50
            if bearish_engulf[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish engulfing OR price rises above weekly EMA50
            if bullish_engulf[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals