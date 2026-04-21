#!/usr/bin/env python3
"""
Hypothesis: 1d weekly Donchian channel breakout with volume confirmation and ADX trend filter.
Longs when price breaks above 10-day Donchian high with ADX>25 and volume>1.5x average;
shorts when price breaks below 10-day Donchian low with ADX>25 and volume>1.5x average.
Exit on price crossing back through 20-day EMA or 2x ATR stop.
Designed for 10-20 trades/year to minimize fee decay while capturing major trends.
Works in bull markets via breakouts and in bear markets via short breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 20-period EMA on weekly close for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 14-period ADX on weekly for trend strength
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    plus_dm = np.zeros_like(high_1w)
    minus_dm = np.zeros_like(high_1w)
    plus_dm[1:] = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                           np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    minus_dm[1:] = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                            np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_1w
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_1w
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1w = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Load daily data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 10-day Donchian channels
    donch_high = pd.Series(high_1d).rolling(window=10, min_periods=10).max().values
    donch_low = pd.Series(low_1d).rolling(window=10, min_periods=10).min().values
    
    # Calculate 20-day EMA for exit
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all daily indicators to lower timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume confirmation: volume spike > 1.5x 20-day average
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    # ATR for stoploss (20-day)
    tr1 = prices['high'].values - prices['low'].values
    tr2 = np.abs(prices['high'].values - np.roll(prices['close'].values, 1))
    tr3 = np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_20_1d_aligned[i]) or np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(adx_1w_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        ema_20_1d_val = ema_20_1d_aligned[i]
        ema_20_1w_val = ema_20_1w_aligned[i]
        adx_val = adx_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: break above Donchian high with weekly trend up and volume
            if (price_high > donch_high_val and 
                ema_20_1w_val > ema_20_1w[max(0, i-1)] if i > 0 else ema_20_1w_val and  # weekly EMA rising
                adx_val > 25 and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: break below Donchian low with weekly trend down and volume
            elif (price_low < donch_low_val and 
                  ema_20_1w_val < ema_20_1w[max(0, i-1)] if i > 0 else ema_20_1w_val and  # weekly EMA falling
                  adx_val > 25 and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: 20-day EMA cross OR ATR-based stoploss
            exit_signal = False
            
            # 20-day EMA exit
            if position == 1 and price_close < ema_20_1d_val:
                exit_signal = True
            elif position == -1 and price_close > ema_20_1d_val:
                exit_signal = True
            
            # ATR-based stoploss (2x ATR from entry level)
            if position == 1:
                # For longs, stop below Donchian low
                if price_close < donch_low_val - 2.0 * atr_val:
                    exit_signal = True
            elif position == -1:
                # For shorts, stop above Donchian high
                if price_close > donch_high_val + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WeeklyDonchian10_EMA20_ADX25_Volume1.5x_ATR2x"
timeframe = "1d"
leverage = 1.0