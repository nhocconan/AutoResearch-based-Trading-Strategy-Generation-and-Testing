#/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_Confirmation_1wTrend
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise - in trending markets it follows price closely, in ranging markets it stays flat.
Combined with RSI for momentum confirmation and 1-week trend filter to avoid counter-trend trades.
Designed for low trade frequency (10-25 trades/year) to minimize fee decay while capturing sustained moves.
Works in both bull (KAMA up + RSI > 50) and bear (KAMA down + RSI < 50) markets.
"""

name = "1d_KAMA_Direction_RSI_Confirmation_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    close_1w = df_1w['close'].values

    # KAMA (Kaufman Adaptive Moving Average) parameters
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    
    # Calculate efficiency ratio
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    er = np.zeros_like(change)
    for i in range(len(change)):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    # Pad ER array to match close length
    er_padded = np.full_like(close, np.nan, dtype=float)
    er_padded[9:] = er  # Start from index 9
    
    # Calculate smoothing constant
    sc = (er_padded * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan, dtype=float)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]

    # RSI (14-period) for momentum confirmation
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad RSI to match length
    rsi_padded = np.full_like(close, np.nan, dtype=float)
    rsi_padded[14:] = rsi

    # 1-week EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after KAMA and RSI warmup
        if np.isnan(kama[i]) or np.isnan(rsi_padded[i]) or np.isnan(ema20_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA rising (trending up) + RSI > 50 (bullish momentum) + price above 1w EMA20 (uptrend)
            if kama[i] > kama[i-1] and rsi_padded[i] > 50 and close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling (trending down) + RSI < 50 (bearish momentum) + price below 1w EMA20 (downtrend)
            elif kama[i] < kama[i-1] and rsi_padded[i] < 50 and close[i] < ema20_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turns down OR RSI becomes bearish OR trend turns down
            if kama[i] < kama[i-1] or rsi_padded[i] < 50 or close[i] < ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turns up OR RSI becomes bullish OR trend turns up
            if kama[i] > kama[i-1] or rsi_padded[i] > 50 or close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals