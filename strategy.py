#!/usr/bin/env python3
"""
Hypothesis: 4h Bollinger Band squeeze breakout with 1d volume spike and ADX trend filter.
Longs when price breaks above upper BB after squeeze (BBW < 20th percentile) with ADX>25 and volume>2x average; shorts when price breaks below lower BB under same conditions.
Exit on price crossing back through middle band (20 SMA) or 2.5x ATR stop.
Uses volatility contraction/expansion to capture breakouts with low false signals.
Designed for 25-40 trades/year to minimize fee decay while capturing high-probability moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for Bollinger Bands and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period Bollinger Bands
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2.0 * std_20
    lower_bb = sma_20 - 2.0 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized bandwidth
    
    # Calculate 20-period percentile rank of BB width (for squeeze detection)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Calculate 14-period ADX for trend filter
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(high_1d)
    plus_dm[1:] = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                           np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm[1:] = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                            np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align Bollinger Bands, BB width percentile, and ADX to 4h timeframe
    sma_20_aligned = align_htf_to_ltf(prices, df_1d, sma_20)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume spike > 2.0x 20-period average
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
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(sma_20_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or np.isnan(bb_width_percentile_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        sma = sma_20_aligned[i]
        upper = upper_bb_aligned[i]
        lower = lower_bb_aligned[i]
        bb_percentile = bb_width_percentile_aligned[i]
        adx_val = adx_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: break above upper BB after squeeze (BBW < 20th percentile) with volume and trend
            if (price_high > upper and 
                bb_percentile < 20 and 
                adx_val > 25 and 
                vol_ratio_val > 2.0):
                signals[i] = 0.25
                position = 1
            # Enter short: break below lower BB after squeeze with volume and trend
            elif (price_low < lower and 
                  bb_percentile < 20 and 
                  adx_val > 25 and 
                  vol_ratio_val > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: middle band cross OR ATR-based stoploss
            exit_signal = False
            
            # Middle band (20 SMA) exit
            if position == 1 and price_close < sma:
                exit_signal = True
            elif position == -1 and price_close > sma:
                exit_signal = True
            
            # ATR-based stoploss (2.5x ATR from entry level)
            if position == 1:
                # For longs, stop below lower band (as proxy for entry area)
                if price_close < lower - 2.5 * atr_val:
                    exit_signal = True
            elif position == -1:
                # For shorts, stop above upper band (as proxy for entry area)
                if price_close > upper + 2.5 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_BollingerSqueeze_Breakout_1dADX25_Volume2x_ATR2.5x"
timeframe = "4h"
leverage = 1.0