#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter + 1-day RSI extremes + 12h ATR-based breakout
# - Choppiness Index > 61.8 indicates ranging market (mean reversion opportunity)
# - RSI < 30 or > 70 on daily timeframe signals overextension
# - Enter mean reversion when price touches 12h Bollinger Bands (20,2) during ranging conditions
# - Exit when RSI returns to neutral (40-60) or opposite extreme
# - Designed for low-frequency trading (<30 trades/year) to minimize fee drag in ranging markets
# - Works in both bull/bear by focusing on mean reversion within ranges rather than directional trends

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE for RSI and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily RSI (14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate daily Bollinger Bands (20, 2)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Align daily indicators to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # Calculate 12h Choppiness Index (14) for regime detection
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index = 100 * log10(sum(ATR)/log10(range)) / log10(14)
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    range_hl = highest_high - lowest_low
    chop = 100 * np.log10(sum_atr) / (np.log10(range_hl) * np.log10(14))
    chop = np.where(range_hl > 0, chop, 50)  # Default to 50 when range is zero
    
    # Calculate 12h Bollinger Bands (20, 2) for entry signals
    sma_12h = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_12h = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb_12h = sma_12h + 2 * std_12h
    lower_bb_12h = sma_12h - 2 * std_12h
    
    # Volume filter: 12h volume > 20-period average
    volume = prices['volume'].values
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in any indicator
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or np.isnan(chop[i]) or 
            np.isnan(upper_bb_12h[i]) or np.isnan(lower_bb_12h[i]) or
            np.isnan(volume_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade in ranging markets (Choppiness > 61.8)
        if chop[i] <= 61.8:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter
        vol_filter = volume[i] > volume_ma_20[i]
        
        price = close[i]
        
        if position == 0:
            # Long signal: price touches lower Bollinger Band (oversold) + RSI < 30
            if price <= lower_bb_12h[i] and rsi_1d_aligned[i] < 30 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short signal: price touches upper Bollinger Band (overbought) + RSI > 70
            elif price >= upper_bb_12h[i] and rsi_1d_aligned[i] > 70 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral (40-60) or touches upper band
            if rsi_1d_aligned[i] >= 40 and rsi_1d_aligned[i] <= 60 or price >= upper_bb_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral (40-60) or touches lower band
            if rsi_1d_aligned[i] >= 40 and rsi_1d_aligned[i] <= 60 or price <= lower_bb_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Choppiness_RSI_Bollinger_MeanReversion"
timeframe = "12h"
leverage = 1.0