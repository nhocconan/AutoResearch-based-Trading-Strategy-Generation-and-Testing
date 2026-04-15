#!/usr/bin/env python3
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
    
    # 1d KAMA for trend direction
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    abs_change = np.sum(np.abs(np.diff(close_1d, k=1)), axis=0)  # placeholder for efficiency ratio calc
    # Simplified ER calculation for KAMA
    price_change = np.abs(close_1d - np.roll(close_1d, 10))
    price_change[0:10] = 0
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder
    # Use EMA as proxy for KAMA trend
    ema_10 = pd.Series(close_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_aligned = align_htf_to_ltf(prices, df_1d, ema_10)
    
    # 1d RSI for overbought/oversold
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # 1d Choppiness Index for regime filter
    atr_1d = np.maximum(high - low,
                        np.maximum(np.abs(high - np.roll(close, 1)),
                                   np.abs(low - np.roll(close, 1))))
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_10_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            continue
            
        # Long: price > KAMA, RSI < 50, chop > 50 (range)
        if close[i] > ema_10_aligned[i] and rsi_aligned[i] < 50 and chop_aligned[i] > 50:
            signals[i] = 0.25
        # Short: price < KAMA, RSI > 50, chop > 50 (range)
        elif close[i] < ema_10_aligned[i] and rsi_aligned[i] > 50 and chop_aligned[i] > 50:
            signals[i] = -0.25
        # Exit: chop < 30 (trend) or opposite signal
        elif chop_aligned[i] < 30 or (i > 0 and ((signals[i-1] == 0.25 and close[i] < ema_10_aligned[i]) or
                                                 (signals[i-1] == -0.25 and close[i] > ema_10_aligned[i]))):
            signals[i] = 0.0
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_KAMA_RSI_Chop_Range"
timeframe = "1d"
leverage = 1.0