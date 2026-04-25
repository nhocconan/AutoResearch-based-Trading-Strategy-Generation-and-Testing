#!/usr/bin/env python3
"""
1d KAMA + RSI + Chop Regime Filter
Hypothesis: Kaufman Adaptive Moving Average (KAMA) identifies trend direction efficiently.
RSI(14) provides momentum confirmation. Chopiness Index (CHOP) filter avoids whipsaws:
- CHOP > 61.8 = ranging market (mean reversion at RSI extremes)
- CHOP < 38.2 = trending market (trend follow with KAMA)
In trending regimes: long when price > KAMA and RSI > 50; short when price < KAMA and RSI < 50.
In ranging regimes: long when RSI < 30 and price > KAMA; short when RSI > 70 and price < KAMA.
Weekly EMA34 trend filter ensures alignment with higher timeframe momentum.
Designed for 1d timeframe to target 7-25 trades/year (30-100 over 4 years) by requiring
confluence of KAMA direction, RSI momentum, and chop regime filter.
Works in bull/bear regimes via adaptive trend filter and regime-specific RSI rules.
"""

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
    
    # Load weekly data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    ema_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # KAMA calculation (adaptive moving average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close - close[10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |diff| over 10 periods
    # Fix array alignment: change has length n-10, volatility has length n-1
    # We'll compute ER using rolling window approach
    close_series = pd.Series(close)
    change_abs = close_series.diff(10).abs()
    volatility_sum = close_series.diff().abs().rolling(window=10, min_periods=10).sum()
    er = change_abs / volatility_sum.replace(0, np.nan)
    
    # Smoothing constants
    fastest = 2 / (2 + 1)   # EMA(2)
    slowest = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fastest - slowest) + slowest) ** 2
    sc = sc.fillna(0.0)  # Handle NaN from division
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Neutral when undefined
    
    # Chopiness Index (CHOP) - measures market choppiness vs trend
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR(14) sum
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max()
    ll = pd.Series(low).rolling(window=14, min_periods=14).min()
    
    # CHOP formula: 100 * log10(atr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero when hh == ll
    range_hl = hh - ll
    chop = 100 * np.log10(atr_sum / range_hl.replace(0, np.nan)) / np.log10(14)
    chop = chop.fillna(50).values  # Neutral when undefined
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all calculations
    start_idx = max(14, 20)  # CHOP, RSI need 14+ periods
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_kama = kama[i]
        curr_rsi = rsi[i]
        curr_chop = chop[i]
        
        # Trend filter from weekly EMA34
        bullish_bias = curr_close > ema_1w_aligned[i]
        bearish_bias = curr_close < ema_1w_aligned[i]
        
        if position == 0:
            # Regime-based entry logic
            if curr_chop > 61.8:  # Ranging market - mean reversion
                # Long when RSI oversold and price above KAMA (bullish bias within range)
                long_entry = (curr_rsi < 30) and (curr_close > curr_kama) and bullish_bias
                # Short when RSI overbought and price below KAMA (bearish bias within range)
                short_entry = (curr_rsi > 70) and (curr_close < curr_kama) and bearish_bias
            else:  # Trending market (CHOP <= 61.8) - trend following
                # Long when price above KAMA and RSI bullish (>50)
                long_entry = (curr_close > curr_kama) and (curr_rsi > 50) and bullish_bias
                # Short when price below KAMA and RSI bearish (<50)
                short_entry = (curr_close < curr_kama) and (curr_rsi < 50) and bearish_bias
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit conditions: regime change or loss of momentum
            if curr_chop > 61.8:  # Went to ranging - exit if RSI normalizes
                if curr_rsi >= 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Still trending - exit if price below KAMA or RSI turns bearish
                if (curr_close < curr_kama) or (curr_rsi < 50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit conditions: regime change or loss of momentum
            if curr_chop > 61.8:  # Went to ranging - exit if RSI normalizes
                if curr_rsi <= 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Still trending - exit if price above KAMA or RSI turns bullish
                if (curr_close > curr_kama) or (curr_rsi > 50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_ChopRegime"
timeframe = "1d"
leverage = 1.0