#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_Pullback_Momentum_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data ONCE before loop for trend direction
    df_4h = get_htf_data(prices, '4h')
    # Get 1d data ONCE before loop for momentum confirmation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h EMA50 for trend direction
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d RSI(14) for momentum confirmation
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 1h ATR(14) for dynamic exit
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Pre-compute hour filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get aligned values
        ema_trend = ema_50_4h_aligned[i]
        rsi_momentum = rsi_1d_aligned[i]
        current_atr = atr[i]
        current_close = prices['close'].iloc[i]
        
        # Skip if any value is NaN
        if np.isnan(ema_trend) or np.isnan(rsi_momentum) or np.isnan(current_atr):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price near 4h EMA50 pullback in uptrend with bullish momentum
            if current_close > ema_trend and rsi_momentum > 50 and current_close < ema_trend * 1.02:
                signals[i] = 0.20
                position = 1
                entry_price = current_close
            # Short: price near 4h EMA50 pullback in downtrend with bearish momentum
            elif current_close < ema_trend and rsi_momentum < 50 and current_close > ema_trend * 0.98:
                signals[i] = -0.20
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: trend break or ATR stop
            if current_close < ema_trend * 0.995:
                signals[i] = 0.0
                position = 0
            elif current_close < entry_price - 2.0 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: trend break or ATR stop
            if current_close > ema_trend * 1.005:
                signals[i] = 0.0
                position = 0
            elif current_close > entry_price + 2.0 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals