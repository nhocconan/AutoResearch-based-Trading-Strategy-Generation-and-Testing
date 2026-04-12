#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # KAMA on daily close
    close_series = pd.Series(close)
    change = np.abs(np.diff(close_series, prepend=close_series.iloc[0]))
    volatility = pd.Series(change).rolling(window=10, min_periods=10).sum().values
    er = np.divide(np.abs(np.diff(close_series, periods=10)), volatility, out=np.zeros_like(volatility), where=volatility!=0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14) on daily
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14) on daily
    atr_list = []
    for i in range(n):
        if i == 0:
            tr = high[i] - low[i]
        else:
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr_list.append(tr)
    atr = np.array(atr_list)
    sum_atr14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr14 / (hh14 - ll14 + 1e-10)) / np.log10(14)
    
    # Weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below weekly EMA34
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        # KAMA direction
        kama_up = kama[i] > kama[i-1]
        kama_down = kama[i] < kama[i-1]
        
        # RSI in normal range (avoid extremes)
        rsi_ok = (rsi[i] >= 30) and (rsi[i] <= 70)
        
        # Chop filter: only trade when chop > 50 (ranging market)
        chop_ok = chop[i] > 50
        
        # Long: KAMA up + RSI OK + chop > 50 + uptrend
        long_signal = kama_up and rsi_ok and chop_ok and uptrend
        # Short: KAMA down + RSI OK + chop > 50 + downtrend
        short_signal = kama_down and rsi_ok and chop_ok and downtrend
        
        # Exit when KAMA reverses
        exit_long = not kama_up
        exit_short = not kama_down
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals