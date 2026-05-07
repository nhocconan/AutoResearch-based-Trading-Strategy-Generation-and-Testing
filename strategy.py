#!/usr/bin/env python3
name = "1d_KAMA_RSI_Chop_Filter"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # KAMA calculation (2-period ER, 30-period smoothing)
    close_series = pd.Series(close)
    change = abs(close_series.diff(2)).values
    volatility = abs(close_series.diff(1)).rolling(window=2, min_periods=2).sum().values
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align weekly KAMA to daily
    kama_1w_series = pd.Series(kama).rolling(window=21, min_periods=21).mean().values
    kama_1w = align_htf_to_ltf(prices, df_1w, kama_1w_series)
    
    # RSI(14) on daily
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14-period)
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(highest_high - lowest_low) / np.log10(14) / np.divide(atr.sum(), 14, out=np.zeros_like(atr), where=atr!=0)
    chop = np.where(cumsum_atr := pd.Series(atr).rolling(window=14, min_periods=14).sum().values != 0,
                    100 * np.log10(highest_high - lowest_low) / np.log10(14) / cumsum_atr,
                    50)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(30, 14, 21)
    
    for i in range(start_idx, n):
        if np.isnan(kama_1w[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter: price above/below KAMA
        uptrend = close[i] > kama_1w[i]
        downtrend = close[i] < kama_1w[i]
        
        if position == 0:
            # Long: KAMA uptrend, RSI oversold, low chop (trending market)
            if uptrend and rsi[i] < 30 and chop[i] < 38.2:
                signals[i] = 0.25
                position = 1
            # Short: KAMA downtrend, RSI overbought, low chop (trending market)
            elif downtrend and rsi[i] > 70 and chop[i] < 38.2:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI overbought or trend change
            if rsi[i] > 70 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI oversold or trend change
            if rsi[i] < 30 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: KAMA trend + RSI mean reversion + Chop filter
# - KAMA(2,30) on weekly timeframe determines primary trend direction
# - RSI(14) < 30 for long, > 70 for short in direction of weekly trend
# - Chop < 38.2 ensures we only trade in trending markets (avoid whipsaws)
# - Works in bull markets: buy dips in uptrend, sell rallies in downtrend
# - Works in bear markets: same logic applies as trend follows price
# - Position size 0.25 balances return and risk
# - Low trade frequency expected due to multiple filters
# - Weekly trend filter prevents counter-trend trading
# - Chop filter avoids ranging markets where RSI mean reversion fails