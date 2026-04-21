#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using 1d Bollinger Bands (20,2) for mean reversion in ranging markets.
In ranging (Choppiness Index > 61.8), buy touches of lower band with bullish rejection,
sell touches of upper band with bearish rejection. Trend filter: price must be between
1d EMA50 and EMA200 to avoid strong trends. Volume must exceed 1.3x 20-period average.
Exit on Bollinger middle band cross or 1.5x ATR stop. Designed for 15-40 trades/year
(60-160 total over 4 years) to minimize fee drag while capturing mean reversion in chop.
Works in bull markets via lower band bounces and in bear markets via upper band rejections.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Bollinger Bands, EMAs, and Choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Bollinger Bands (20,2)
    bb_length = 20
    bb_mult = 2.0
    basis = pd.Series(close_1d).rolling(window=bb_length, min_periods=bb_length).mean().values
    dev = bb_mult * pd.Series(close_1d).rolling(window=bb_length, min_periods=bb_length).std().values
    upper_band = basis + dev
    lower_band = basis - dev
    
    # Calculate 1d EMAs for trend filter (EMA50 and EMA200)
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 1d Choppiness Index (14) for regime filter
    chop_length = 14
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=chop_length, min_periods=chop_length).sum().values
    hh = pd.Series(high_1d).rolling(window=chop_length, min_periods=chop_length).max().values
    ll = pd.Series(low_1d).rolling(window=chop_length, min_periods=chop_length).min().values
    # Avoid division by zero
    range_max_min = hh - ll
    range_max_min = np.where(range_max_min == 0, 1e-10, range_max_min)
    chop = 100 * np.log10(atr_sum / range_max_min) / np.log10(chop_length)
    
    # Align 1d indicators to 12h timeframe (wait for 1d bar to close)
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    basis_aligned = align_htf_to_ltf(prices, df_1d, basis)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation (volume spike > 1.3x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    # ATR for stoploss (20-period)
    tr1 = prices['high'].values - prices['low'].values
    tr2 = np.abs(prices['high'].values - np.roll(prices['close'].values, 1))
    tr3 = np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if indicators not ready
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(basis_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_low = prices['low'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_open = prices['open'].iloc[i]
        upper = upper_band_aligned[i]
        lower = lower_band_aligned[i]
        basis_val = basis_aligned[i]
        ema_50_val = ema_50_aligned[i]
        ema_200_val = ema_200_aligned[i]
        chop_val = chop_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        # Range regime: Choppiness > 61.8 indicates ranging market
        is_ranging = chop_val > 61.8
        
        if position == 0 and is_ranging:
            # Enter long: price touches lower band with bullish rejection, price between EMAs
            if (price_low <= lower * 1.001 and  # Allow 0.1% tolerance for touch
                price_close > price_open and  # Bullish candle
                price_close > ema_50_val and 
                price_close < ema_200_val and 
                vol_ratio_val > 1.3):
                signals[i] = 0.25
                position = 1
            # Enter short: price touches upper band with bearish rejection, price between EMAs
            elif (price_high >= upper * 0.999 and  # Allow 0.1% tolerance for touch
                  price_close < price_open and  # Bearish candle
                  price_close < ema_50_val and 
                  price_close > ema_200_val and 
                  vol_ratio_val > 1.3):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: Bollinger middle band cross OR ATR-based stoploss
            exit_signal = False
            
            # Bollinger middle band exit
            if position == 1 and price_close < basis_val:
                exit_signal = True
            elif position == -1 and price_close > basis_val:
                exit_signal = True
            
            # ATR-based stoploss (1.5x ATR from entry level)
            if position == 1:
                entry_approx = lower  # Entered near lower band
                if price_close < entry_approx - 1.5 * atr_val:
                    exit_signal = True
            elif position == -1:
                entry_approx = upper  # Entered near upper band
                if price_close > entry_approx + 1.5 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_BollingerBands_Reversion_1dEMA50_200_Chop_Volume_ATR"
timeframe = "12h"
leverage = 1.0