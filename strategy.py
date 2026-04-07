#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d KAMA + RSI + Chop Filter
# Hypothesis: KAMA adapts to market noise, reducing whipsaw in sideways markets.
# RSI identifies overbought/oversold conditions, while Choppiness Index filters
# for trending regimes. This combination aims to capture trends with fewer
# false signals, suitable for both bull and bear markets.
# Target: 10-25 trades/year to minimize fee drag on 1d timeframe.
name = "1d_kama_rsi_chop_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily close
    # ER = |Close - Close(p)| / sum(|Close - Close-1|) for p periods
    # SC = [ER * (fastest - slowest) + slowest]^2
    # KAMA = KAMA(prev) + SC * (Close - KAMA(prev))
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    close_s = pd.Series(close)
    change = abs(close_s.diff(1)).values
    volatility = pd.Series(change).rolling(window=10, min_periods=10).sum().values
    net_change = abs(close_s.diff(10)).values
    er = np.where(volatility != 0, net_change / volatility, 0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values
    
    # Calculate Choppiness Index(14)
    # CHOP = 100 * log10(sum(ATR) / (max(HH) - min(LL))) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(14)
    
    # 1-week EMA(20) for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=20, adjust=False).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(chop[i]) or np.isnan(weekly_ema_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below KAMA or RSI overbought
            if close[i] < kama[i] or rsi_values[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price crosses above KAMA or RSI oversold
            if close[i] > kama[i] or rsi_values[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Only trade in trending markets (CHOP < 61.8)
            if chop[i] < 61.8:
                # Enter long: price above KAMA and RSI recovering from oversold
                if close[i] > kama[i] and rsi_values[i] > 30 and rsi_values[i] < 50:
                    position = 1
                    signals[i] = 0.25
                # Enter short: price below KAMA and RSI declining from overbought
                elif close[i] < kama[i] and rsi_values[i] < 70 and rsi_values[i] > 50:
                    position = -1
                    signals[i] = -0.25
    
    return signals