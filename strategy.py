#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h RSI(14) for mean reversion in ranging markets and 1d trend filter.
# Uses 4h RSI(14) with extreme levels (RSI<30 for long, RSI>70 for short) in ranging markets.
# 1d EMA(50) trend filter: only take long when price > 1d EMA50, short when price < 1d EMA50.
# Session filter: 08-20 UTC to avoid low-volume Asian session noise.
# Designed for low trade frequency (~15-37/year on 1h) to minimize fee drag.
# Works in bull markets via trend-following longs and in bear markets via mean-reversion shorts.
# Focus on BTC/ETH as primary targets.

name = "1h_4hRSI14_1dEMA50_Trend_MeanReversion_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours for 08-20 UTC filter
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for RSI calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate 4h RSI(14)
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_4h = 100 - (100 / (1 + rs))
    
    # Align 4h RSI to 1h timeframe
    rsi_14_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_14_4h)
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 1h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for dynamic stoploss on 1h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 100  # warmup for RSI(14) and EMA(50)
    
    for i in range(start_idx, n):
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            continue
            
        curr_close = close[i]
        curr_rsi = rsi_14_4h_aligned[i]
        curr_ema = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Mean reversion entries with trend filter
            if curr_rsi < 30 and curr_close > curr_ema:  # Oversold + uptrend filter
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            elif curr_rsi > 70 and curr_close < curr_ema:  # Overbought + downtrend filter
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Take profit: RSI returns to neutral (50) or reaches 1.5x ATR profit
            elif curr_rsi >= 50 or curr_close > entry_price + 1.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Take profit: RSI returns to neutral (50) or reaches 1.5x ATR profit
            elif curr_rsi <= 50 or curr_close < entry_price - 1.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals