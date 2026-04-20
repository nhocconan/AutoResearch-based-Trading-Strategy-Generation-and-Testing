#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Choppiness_KAMA_Trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # === 1d: Core price data ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d: KAMA (Kaufman Adaptive Moving Average) ===
    close_series = pd.Series(close)
    # Efficiency Ratio (ER)
    change = abs(close_series - close_series.shift(10))
    volatility = abs(close_series.diff()).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === 1d: RSI(14) ===
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # === 1d: Choppiness Index (14) ===
    atr_list = []
    for i in range(n):
        if i == 0:
            tr = 0
        else:
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr = max(tr1, tr2, tr3)
        atr_list.append(tr)
    atr = np.array(atr_list)
    # True Range sum over period
    tr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over period
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Choppiness Index
    chop = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(14)
    chop = np.where((hh - ll) == 0, 50, chop)  # avoid division by zero
    
    # === 1w: EMA34 for trend filter ===
    close_1w = pd.Series(df_1w['close'].values)
    ema34_1w = close_1w.ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get values
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        close_val = close[i]
        ema34_1w_val = ema34_1w_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(chop_val) or 
            np.isnan(ema34_1w_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA upward + RSI > 50 + Choppiness < 61.8 (trending) + price > weekly EMA34
            if (close_val > kama_val and rsi_val > 50 and chop_val < 61.8 and 
                close_val > ema34_1w_val):
                signals[i] = 0.25
                position = 1
            
            # Short: KAMA downward + RSI < 50 + Choppiness < 61.8 (trending) + price < weekly EMA34
            elif (close_val < kama_val and rsi_val < 50 and chop_val < 61.8 and 
                  close_val < ema34_1w_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA downward OR RSI < 40 OR Choppiness > 61.8 (choppy)
            if (close_val < kama_val or rsi_val < 40 or chop_val > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA upward OR RSI > 60 OR Choppiness > 61.8 (choppy)
            if (close_val > kama_val or rsi_val > 60 or chop_val > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals