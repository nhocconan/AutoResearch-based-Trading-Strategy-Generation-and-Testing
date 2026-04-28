#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with RSI(14) mean reversion and choppiness regime filter.
# Enter long when KAMA turns up, RSI < 40 (oversold), and choppiness > 61.8 (ranging market).
# Enter short when KAMA turns down, RSI > 60 (overbought), and choppiness > 61.8.
# Exit when RSI crosses 50 (mean reversion complete) or choppiness < 38.2 (trending market).
# Uses discrete position sizing (0.25) to balance return and drawdown.
# Target: 50-100 total trades over 4 years (12-25/year).
# KAMA adapts to market noise, RSI captures mean reversion in ranges, chop filter avoids false signals in trends.
# Works in bull markets (buy dips in ranges) and bear markets (sell rallies in ranges).

name = "1d_KAMA_RSI_Chop_MeanReversion_v1"
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
    
    # Calculate KAMA(10, 2, 30)
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start after first 10 periods
    for i in range(10, n):
        if np.isnan(kama[i-1]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index(14)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Choppiness
    chop = np.where((hh - ll) != 0, 
                    100 * np.log10(atr_sum / (hh - ll)) / np.log10(14), 
                    50)
    
    # Get 1w data for higher timeframe trend filter (optional, for confirmation)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 50:
        close_1w = df_1w['close'].values
        ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
        ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
        # Use 1w EMA20 as trend filter: only take longs in uptrend, shorts in downtrend
        trend_filter = True  # We'll use this below
    else:
        ema_20_1w_aligned = np.full(n, np.nan)
        trend_filter = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(kama[i-1]) or 
            np.isnan(rsi[i]) or np.isnan(chop[i]) or
            (trend_filter and np.isnan(ema_20_1w_aligned[i]))):
            signals[i] = 0.0
            continue
        
        # KAMA direction: comparing current to previous
        kama_up = kama[i] > kama[i-1]
        kama_down = kama[i] < kama[i-1]
        
        # RSI levels
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_exit = 45 <= rsi[i] <= 55  # Exit zone around 50
        
        # Choppiness regime: only trade in ranging markets
        chop_ranging = chop[i] > 61.8
        chop_trending = chop[i] < 38.2
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: KAMA up, RSI oversold, ranging market
            if kama_up and rsi_oversold and chop_ranging:
                signals[i] = 0.25
                position = 1
            # Short entry: KAMA down, RSI overbought, ranging market
            elif kama_down and rsi_overbought and chop_ranging:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit
            # Exit conditions: KAMA down, RSI back to neutral, or trending market
            if kama_down or rsi_exit or chop_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit
            # Exit conditions: KAMA up, RSI back to neutral, or trending market
            if kama_up or rsi_exit or chop_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals