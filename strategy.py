#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Choppiness Index with RSI Mean Reversion
# Uses Choppiness Index (14-period) to identify ranging markets (CHOP > 61.8)
# RSI (14) for mean reversion: long when RSI < 30, short when RSI > 70
# Only trade when market is ranging (high chop) to avoid trending markets
# Weekly trend filter: only trade long when price > weekly EMA(50), short when price < weekly EMA(50)
# Designed for mean reversion in ranging markets with trend filter to avoid major losses
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA (50) for trend direction
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate Choppiness Index (14-period) on daily data
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high) - min(low))))
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # first bar
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    hh_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    range_14[range_14 == 0] = 1e-10
    
    chop = 100 * np.log10(atr_14 * 14 / np.log10(14)) / np.log10(range_14)
    chop = np.where(range_14 > 0, chop, 50)  # default to 50 when range is zero
    
    # Calculate RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Avoid division by zero
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 14  # for RSI and Chop
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Only trade in ranging markets (high chop)
        is_ranging = chop[i] > 61.8
        
        if position == 0:
            # Long: RSI oversold in ranging market and above weekly EMA
            if rsi[i] < 30 and is_ranging and price > ema_1w_aligned[i]:
                position = 1
                signals[i] = position_size
            # Short: RSI overbought in ranging market and below weekly EMA
            elif rsi[i] > 70 and is_ranging and price < ema_1w_aligned[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI overbought or breaks below weekly EMA
            if rsi[i] > 70 or price < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI oversold or breaks above weekly EMA
            if rsi[i] < 30 or price > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Choppiness_RSI_MeanReversion"
timeframe = "1d"
leverage = 1.0