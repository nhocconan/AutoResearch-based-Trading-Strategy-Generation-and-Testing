#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter_v1
Hypothesis: Use 1d KAMA (adaptive trend) as primary trend filter, combined with 1d RSI for momentum confirmation and 1d Choppiness Index for regime filtering. Enter long when KAMA trending up, RSI > 50, and market is trending (CHOP < 38.2). Enter short when KAMA trending down, RSI < 50, and CHOP < 38.2. Exit on opposite signal or when CHOP > 61.8 (range regime). This strategy aims to capture strong trends while avoiding whipsaws in ranging markets, suitable for both bull and bear markets. Discrete position sizing (0.25) minimizes fee churn. Target 30-80 trades over 4 years (7-20/year) to stay within fee drag limits for 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for longer-term trend context)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1d KAMA (adaptive trend) ===
    close = prices['close'].values
    # Calculate Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close[t] - close[t-1]| over 10 periods
    # Avoid division by zero
    er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
    # Smoothing constants: fastest EMA = 2/(2+1) = 0.67, slowest = 2/(30+1) = 0.0645
    sc = (er * (0.67 - 0.0645) + 0.0645) ** 2
    # Calculate KAMA
    kama = np.full_like(close, np.nan, dtype=float)
    kama[9] = close[9]  # start after 10 periods
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === 1d RSI (14-period) ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain, dtype=float), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1d Choppiness Index (14-period) ===
    high = prices['high'].values
    low = prices['low'].values
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    max_hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Choppiness Index
    chop = 100 * np.log10(tr_sum / (max_hh - min_ll)) / np.log10(14)
    # Avoid division by zero and invalid values
    chop = np.where((max_hh - min_ll) > 0, chop, 50.0)
    
    # Align 1w EMA50 for longer-term trend filter (optional context)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after KAMA/RSI/CHOP warmup
        # Skip if indicators not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        ema_50_1w_val = ema_50_1w_aligned[i]
        
        # Trend direction from KAMA: comparing to prior value
        kama_rising = kama_val > kama[i-1] if i > 0 else False
        kama_falling = kama_val < kama[i-1] if i > 0 else False
        
        # Regime filter: trending market (CHOP < 38.2) vs ranging (CHOP > 61.8)
        trending_market = chop_val < 38.2
        ranging_market = chop_val > 61.8
        
        if position == 0:
            # Long: KAMA rising, RSI > 50, trending market
            long_condition = kama_rising and (rsi_val > 50) and trending_market
            # Short: KAMA falling, RSI < 50, trending market
            short_condition = kama_falling and (rsi_val < 50) and trending_market
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: opposite signal or ranging market
            if position == 1:
                # Exit long: KAMA falling OR RSI < 50 OR ranging market
                if kama_falling or (rsi_val < 50) or ranging_market:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: KAMA rising OR RSI > 50 OR ranging market
                if kama_rising or (rsi_val > 50) or ranging_market:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0