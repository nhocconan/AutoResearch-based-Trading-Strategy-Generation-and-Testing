#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_Volume_Regime_ATRStop
Hypothesis: Daily Camarilla R1/S1 breakout with volume confirmation (>1.5x 20-day volume MA) and choppiness regime filter (CHOP > 61.8 for mean reversion, CHOP < 38.2 for trend following). Uses 1w HTF for trend filter (price > EMA50 for longs, < EMA50 for shorts). ATR-based stoploss via signal=0 when price moves against position by 2.0*ATR. Designed for low trade frequency (<150 total 1d trades) to minimize fee drag and work in both bull/bear markets via regime adaptation.
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
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Camarilla pivot levels (based on previous day)
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    camarilla_high = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_low = close_1d - 1.1 * (high_1d - low_1d) / 12
    # Shift by 1 to use previous day's levels
    camarilla_high = np.roll(camarilla_high, 1)
    camarilla_low = np.roll(camarilla_low, 1)
    camarilla_high[0] = np.nan
    camarilla_low[0] = np.nan
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index (14-period)
    chop_sum = tr.rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(chop_sum / (highest_high - lowest_low)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_high[i]) or np.isnan(camarilla_low[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(chop[i])
            or np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        vol = volume_1d[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        # Regime detection
        is_choppy = chop[i] > 61.8  # mean reversion regime
        is_trending = chop[i] < 38.2  # trend following regime
        
        if position == 0:
            # Long: Camarilla R1 breakout + volume + trend filter (in uptrend or choppy market)
            if price > camarilla_high[i] and vol_ok and (price > ema_50_1w_aligned[i] or is_choppy):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Camarilla S1 breakdown + volume + trend filter (in downtrend or choppy market)
            elif price < camarilla_low[i] and vol_ok and (price < ema_50_1w_aligned[i] or is_choppy):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: Camarilla S1 breakdown or loss of volume/momentum
            elif price < camarilla_low[i] or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: Camarilla R1 breakout or loss of volume/momentum
            elif price > camarilla_high[i] or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_Volume_Regime_ATRStop"
timeframe = "1d"
leverage = 1.0