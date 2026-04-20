#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter + 1-day RSI mean reversion
# In high chop (CHOP > 61.8): mean revert at RSI extremes (RSI<30 long, RSI>70 short)
# In low chop (CHOP < 38.2): follow trend (price > EMA50 long, price < EMA50 short)
# Uses 1-day RSI and 12h EMA50 for signals, 12h Choppiness Index for regime filter
# Designed to work in both bull and bear markets by adapting to market regime
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 14-period RSI on daily timeframe
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Load 12h data for EMA50 and Choppiness Index
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate 12h EMA50 for trend following
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h Choppiness Index (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(rsi_1d_aligned[i]) or np.isnan(ema50[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime classification
        is_choppy = chop[i] > 61.8  # Range market
        is_trending = chop[i] < 38.2  # Trending market
        
        price = close[i]
        
        if position == 0:
            # Enter long conditions
            long_signal = False
            if is_choppy:
                # Mean reversion in chop: RSI oversold
                if rsi_1d_aligned[i] < 30:
                    long_signal = True
            elif is_trending:
                # Trend following: price above EMA50
                if price > ema50[i]:
                    long_signal = True
            
            # Enter short conditions
            short_signal = False
            if is_choppy:
                # Mean reversion in chop: RSI overbought
                if rsi_1d_aligned[i] > 70:
                    short_signal = True
            elif is_trending:
                # Trend following: price below EMA50
                if price < ema50[i]:
                    short_signal = True
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: opposite signal or regime change against position
            exit_signal = False
            if is_choppy and rsi_1d_aligned[i] > 50:  # RSI mean reversion exit
                exit_signal = True
            elif is_trending and price < ema50[i]:  # Trend follow exit
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: opposite signal or regime change against position
            exit_signal = False
            if is_choppy and rsi_1d_aligned[i] < 50:  # RSI mean reversion exit
                exit_signal = True
            elif is_trending and price > ema50[i]:  # Trend follow exit
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_ChopRegime_RSI_EMA50"
timeframe = "12h"
leverage = 1.0