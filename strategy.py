#!/usr/bin/env python3
# 1d_1w_kama_rsi_volume_v1
# Strategy: 1d KAMA trend direction + RSI + volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: KAMA adapts to market noise, providing reliable trend direction.
# In trending markets, KAMA follows price with less lag than traditional MAs.
# Combine with RSI for momentum confirmation and volume to filter weak moves.
# Designed for very low frequency (<10 trades/year) to minimize fee decay in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(20) for trend filter (slower trend)
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # KAMA calculation (adaptive moving average)
    # Efficiency Ratio (ER) = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(close - np.roll(close, 10))
    # Avoid division by zero - add small epsilon
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0) if len(close) > 1 else 1
    # Proper ER calculation using rolling window
    change_series = pd.Series(change)
    volatility_series = pd.Series(np.abs(np.diff(close, prepend=close[0])))
    er = change_series.rolling(window=10, min_periods=1).sum() / \
         volatility_series.rolling(window=10, min_periods=1).sum().replace(0, 1e-10)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan, dtype=float)
    kama[9] = close[:10].mean()  # Start after first 10 periods
    for i in range(10, n):
        if not np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI calculation
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi.iloc[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Trend filters: price above/below KAMA and 1w EMA20
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        price_above_1w_ema = close[i] > ema_20_1w_aligned[i]
        price_below_1w_ema = close[i] < ema_20_1w_aligned[i]
        
        # Entry logic: strong momentum + volume + trend alignment
        if (rsi.iloc[i] > 60 and  # Strong bullish momentum
            price_above_kama and price_above_1w_ema and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.20
        elif (rsi.iloc[i] < 40 and  # Strong bearish momentum
              price_below_kama and price_below_1w_ema and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.20
        # Exit: momentum weakening or trend change
        elif position == 1 and (rsi.iloc[i] < 50 or not price_above_kama):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi.iloc[i] > 50 or not price_below_kama):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals