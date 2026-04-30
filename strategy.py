#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h ADX trend strength filter with 1d EMA200 direction and RSI(2) mean reversion entries
# Uses 1d EMA200 for bull/bear regime, ADX(14) > 25 to confirm trending conditions,
# and RSI(2) < 10 for long or > 90 for short in direction of 1d trend.
# Session filter (08-20 UTC) reduces noise. Discrete sizing 0.20 limits fee drag.
# Target: 60-150 total trades over 4 years (15-37/year) by requiring confluence of trend, momentum, and extreme RSI.

name = "1h_ADXTrend_1dEMA200_RSI2_v1"
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
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1d EMA(200) for regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 1h ADX(14) for trend strength
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1h RSI(2) for mean reversion entries
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 200, 14, 2)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(adx[i]) or
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_200_1d = ema_200_1d_aligned[i]
        curr_adx = adx[i]
        curr_rsi = rsi[i]
        
        if position == 0:  # Flat - look for new entries
            # Require ADX > 25 for trending market
            if curr_adx > 25:
                # Bullish regime: price above 1d EMA200
                if curr_close > curr_ema_200_1d:
                    # Long entry: RSI(2) extremely oversold
                    if curr_rsi < 10:
                        signals[i] = 0.20
                        position = 1
                        entry_price = curr_close
                # Bearish regime: price below 1d EMA200
                elif curr_close < curr_ema_200_1d:
                    # Short entry: RSI(2) extremely overbought
                    if curr_rsi > 90:
                        signals[i] = -0.20
                        position = -1
                        entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit: RSI returns to neutral (50) or loses 1d uptrend
            if (curr_rsi > 50 or  # RSI returned to neutral
                curr_close < curr_ema_200_1d):  # Lost 1d uptrend
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral (50) or loses 1d downtrend
            if (curr_rsi < 50 or  # RSI returned to neutral
                curr_close > curr_ema_200_1d):  # Lost 1d downtrend
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals