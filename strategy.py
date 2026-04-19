#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_Trend_Pullback_RSI_V1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Hour filter: 8-20 UTC (already datetime64[ms], use index.hour)
    hours = prices.index.hour
    
    # 4h trend: EMA34 on close
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # 1d trend filter: price vs EMA50 (avoid counter-trend in strong trends)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1h RSI(14) for pullback entries
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)
    
    for i in range(start_idx, n):
        # Session filter: 8-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
            
        if np.isnan(ema_34_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
            
        price = close[i]
        ema_4h = ema_34_4h_aligned[i]
        ema_1d = ema_50_1d_aligned[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Long: uptrend (price > 4h EMA34 AND price > 1d EMA50) + RSI pullback (<40)
            if price > ema_4h and price > ema_1d and rsi_val < 40:
                signals[i] = 0.20
                position = 1
            # Short: downtrend (price < 4h EMA34 AND price < 1d EMA50) + RSI bounce (>60)
            elif price < ema_4h and price < ema_1d and rsi_val > 60:
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long exit: RSI overbought (>70) or trend break (price < 4h EMA34)
            if rsi_val > 70 or price < ema_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short exit: RSI oversold (<30) or trend break (price > 4h EMA34)
            if rsi_val < 30 or price > ema_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals