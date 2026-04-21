#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_Volume_Regime_ATRStop_V1
Hypothesis: Daily Camarilla pivot R1/S1 breakout with volume confirmation (>1.5x 20-day volume MA) and choppiness regime filter (CHOP > 61.8 for mean reversion, CHOP < 38.2 for trend following). Uses 1w HTF for trend filter (price > EMA50 for longs, < EMA50 for shorts). ATR-based stoploss via signal=0 when price moves against position by 2.0*ATR. Designed for very low trade frequency (<100 total 1d trades) to minimize fee drag and work in both bull/bear markets via regime adaptation. Focus on BTC/ETH with SOL as secondary.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for EMA trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w EMA50 for trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 1d Indicators (primary timeframe) ===
    # Need to get 1d data from prices (which is 1d timeframe per experiment)
    # Since timeframe is 1d, prices already contains 1d data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels from previous day
    # For daily data, we use previous day's OHLC
    # Shift by 1 to get previous day's values
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev = np.roll(close, 1)
    # Set first value to NaN since no previous day
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    # Camarilla levels
    pivot = (high_prev + low_prev + close_prev) / 3
    range_1d = high_prev - low_prev
    r1 = pivot + (range_1d * 1.1 / 12)
    s1 = pivot - (range_1d * 1.1 / 12)
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index (14-period)
    chop_sum = tr.rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(chop_sum / (highest_high - lowest_low)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(chop[i])
            or np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        # Regime detection
        is_choppy = chop[i] > 61.8  # mean reversion regime
        is_trending = chop[i] < 38.2  # trend following regime
        
        if position == 0:
            # Long: Camarilla S1 breakout + volume + trend filter (in uptrend or choppy market)
            if price > s1[i] and vol_ok and (price > ema_50_1w_aligned[i] or is_choppy):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Camarilla R1 breakdown + volume + trend filter (in downtrend or choppy market)
            elif price < r1[i] and vol_ok and (price < ema_50_1w_aligned[i] or is_choppy):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: price back below S1 or loss of volume/momentum
            elif price < s1[i] or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: price back above R1 or loss of volume/momentum
            elif price > r1[i] or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_Volume_Regime_ATRStop_V1"
timeframe = "1d"
leverage = 1.0