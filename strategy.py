#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction + RSI(14) + chop regime filter
# Uses 1d primary timeframe for KAMA trend and RSI signals
# Weekly chop regime (CHOP > 61.8) filters for ranging markets where mean reversion works
# KAMA adapts to market noise, reducing false signals in choppy conditions
# RSI(14) < 30 for long, > 70 for short with trend filter (price > KAMA for longs, < KAMA for shorts)
# Discrete position sizing (0.25) balances profit potential with fee drag minimization
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Works in both bull and bear markets by adapting to regime - mean reversion in chop, trend following in strong trends

name = "1d_KAMA_RSI_ChopRegime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for chop regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1d KAMA(14, 2, 30)
    close_1d = pd.Series(close)
    # Efficiency Ratio
    change = abs(close_1d - close_1d.shift(14))
    volatility = close_1d.diff().abs().rolling(window=14, min_periods=14).sum()
    er = change / volatility.replace(0, np.nan)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close_1d, dtype=float)
    kama[0] = close_1d.iloc[0]
    for i in range(1, len(close_1d)):
        if not np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1] + sc.iloc[i] * (close_1d.iloc[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    kama_values = kama
    kama_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama_values)
    
    # Calculate 1d RSI(14)
    delta = close_1d.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), rsi_values)
    
    # Calculate 1w Chop Index(14)
    high_1w = pd.Series(df_1w['high'])
    low_1w = pd.Series(df_1w['low'])
    close_1w = pd.Series(df_1w['close'])
    # True Range
    tr1 = high_1w - low_1w
    tr2 = abs(high_1w - close_1w.shift(1))
    tr3 = abs(low_1w - close_1w.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).sum()
    # Chop formula
    highest_high = high_1w.rolling(window=14, min_periods=14).max()
    lowest_low = low_1w.rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr / (highest_high - lowest_low)) / np.log10(14)
    chop_values = chop.values
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_values, additional_delay_bars=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Chop regime filter: only trade when CHOP > 61.8 (ranging market)
        in_chop_regime = chop_aligned[i] > 61.8
        
        if position == 0:  # Flat - look for new entries
            # RSI mean reversion signals
            rsi_oversold = rsi_aligned[i] < 30
            rsi_overbought = rsi_aligned[i] > 70
            
            # KAMA trend filter for confirmation
            price_above_kama = close[i] > kama_aligned[i]
            price_below_kama = close[i] < kama_aligned[i]
            
            if rsi_oversold and price_above_kama and in_chop_regime:
                signals[i] = 0.25
                position = 1
            elif rsi_overbought and price_below_kama and in_chop_regime:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: RSI > 50 or price < KAMA
            if rsi_aligned[i] > 50 or close[i] < kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: RSI < 50 or price > KAMA
            if rsi_aligned[i] < 50 or close[i] > kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals