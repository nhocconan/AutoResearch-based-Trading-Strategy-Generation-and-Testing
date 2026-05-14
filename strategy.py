#!/usr/bin/env python3
# Hypothesis: 1d KAMA trend direction with 1w EMA34 filter and RSI(14) mean reversion entries.
# Uses Kaufman Adaptive Moving Average (KAMA) on 1d to determine primary trend,
# 1-week EMA34 as higher-timeframe trend confirmation,
# and RSI(14) < 30 for longs / > 70 for shorts as mean-reversion entries in the direction of trend.
# Discrete position sizing (0.0, ±0.25) minimizes fee churn.
# Designed to capture trend-following mean-reversion pulls in both bull and bear markets.
# Targets 15-25 trades/year per symbol.

name = "1d_KAMA_Trend_1wEMA34_RSI_MR_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # --- 1d Indicators (LTF) ---
    # KAMA(10, 2, 30) - adaptive trend
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else np.abs(np.diff(close, prepend=close[0]))
    # Correct volatility calculation: sum of absolute changes over lookback
    volatility = pd.Series(close).rolling(window=10, min_periods=10).apply(lambda x: np.sum(np.abs(np.diff(x))), raw=True).values
    volatility[0:10] = np.nan  # not enough data
    er = np.where(volatility > 0, np.abs(np.diff(close, prepend=close[0])) / volatility, 0)
    er[0] = 0  # first value
    # Smooth ER
    er_smoothed = pd.Series(er).ewm(span=10, adjust=False, min_periods=10).mean().values
    sc = (er_smoothed * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14) for mean reversion entries
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    # EMA34 on 1w
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(10, n):  # start after KAMA warmup
        # Skip if missing data
        if (np.isnan(kama[i]) or
            np.isnan(rsi[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend: price > KAMA = uptrend, price < KAMA = downtrend
        uptrend = close[i] > kama[i]
        downtrend = close[i] < kama[i]
        
        # HTF trend filter: only trade if 1w EMA34 agrees with 1d KAMA trend
        if uptrend and close[i] > ema_34_1w_aligned[i]:
            # Uptrend confirmed: look for RSI oversold longs
            if position == 0 and rsi[i] < 30:
                signals[i] = 0.25
                position = 1
            elif position == 1:
                # Exit long on RSI overbought or trend change
                if rsi[i] > 70 or close[i] < kama[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short if uptrend confirmed
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
        elif downtrend and close[i] < ema_34_1w_aligned[i]:
            # Downtrend confirmed: look for RSI overbought shorts
            if position == 0 and rsi[i] > 70:
                signals[i] = -0.25
                position = -1
            elif position == -1:
                # Exit short on RSI oversold or trend change
                if rsi[i] < 30 or close[i] > kama[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            elif position == 1:
                # Exit long if downtrend confirmed
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
        else:
            # No clear trend or HTF disagreement: stay flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                # Exit long if trend broken
                signals[i] = 0.0
                position = 0
            elif position == -1:
                # Exit short if trend broken
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
    
    return signals