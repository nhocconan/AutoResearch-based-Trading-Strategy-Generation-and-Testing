#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Choppiness Index regime filter + 1-week Williams %R reversal
# Long when weekly Williams %R < -80 (oversold) + daily Choppiness > 61.8 (ranging market)
# Short when weekly Williams %R > -20 (overbought) + daily Choppiness > 61.8 (ranging market)
# Exit when price crosses 24-period EMA in opposite direction
# Stoploss at 2.5 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses daily Choppiness for regime detection and weekly Williams %R for mean reversion
# Target: 50-150 total trades over 4 years (12-37/year)

name = "12h_choppiness_williamsr_reversal_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily data for Choppiness Index (regime filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Weekly data for Williams %R (mean reversion signal)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate Daily Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = t1[0]
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # True Range sum over 14 periods
    tr_sum = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(tr_sum / (hh_1d - ll_1d)) / log10(14)
    # Avoid division by zero
    range_1d = hh_1d - ll_1d
    chop = np.where(range_1d > 0, 100 * np.log10(tr_sum / range_1d) / np.log10(14), 50)
    chop = np.where(np.isnan(chop), 50, chop)
    chop = np.where(np.isinf(chop), 50, chop)
    chop_1d = chop
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate Weekly Williams %R (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Highest high and lowest low over 14 periods
    hh_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Williams %R: (highest high - close) / (highest high - lowest low) * -100
    wr = np.where((hh_1w - ll_1w) != 0, ((hh_1w - close_1w) / (hh_1w - ll_1w)) * -100, -50)
    wr = np.where(np.isnan(wr), -50, wr)
    wr_1w = wr
    wr_1w_aligned = align_htf_to_ltf(prices, df_1w, wr_1w)
    
    # 24-period EMA for exit
    ema_24 = pd.Series(close).ewm(span=24, adjust=False, min_periods=24).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(24, n):
        # Skip if required data not available
        if (np.isnan(chop_1d_aligned[i]) or np.isnan(wr_1w_aligned[i]) or 
            np.isnan(ema_24[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below 24-period EMA
            elif close[i] < ema_24[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above 24-period EMA
            elif close[i] > ema_24[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Williams %R extremes + Choppiness regime filter
            # Regime filter: Choppiness > 61.8 (ranging market)
            chop_filter = chop_1d_aligned[i] > 61.8
            
            # Long: Williams %R < -80 (oversold) + ranging market
            if wr_1w_aligned[i] < -80 and chop_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Williams %R > -20 (overbought) + ranging market
            elif wr_1w_aligned[i] > -20 and chop_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals