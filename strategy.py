#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h price action combined with 12h/1d regime filters and volume confirmation.
# Uses 12h EMA50 for trend direction (long when price > EMA50, short when price < EMA50).
# 1d ADX > 25 indicates trending market (use trend-following logic), ADX <= 25 indicates ranging (use mean-reversion at Bollinger Bands).
# Volume confirmation requires current volume > 1.3x 20-period average to filter weak signals.
# Designed to work in both bull and bear markets by adapting to regime via ADX.
# Targets 20-40 trades/year with strict entry conditions to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data for EMA trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 50-period EMA on 12h data
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Load 1d data for ADX and Bollinger Bands (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period) on 1d data
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    tr = np.maximum(high_1d[1:] - low_1d[1:], np.maximum(np.abs(high_1d[1:] - high_1d[:-1]), np.abs(low_1d[1:] - low_1d[:-1])))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx = np.concatenate([np.full(14, np.nan), adx])  # align with original length
    
    # Calculate Bollinger Bands (20, 2) on 1d data
    bb_middle = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Align 1d indicators to 6h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    bb_middle_aligned = align_htf_to_ltf(prices, df_1d, bb_middle)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(bb_middle_aligned[i]) or 
            np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema_val = ema_12h_aligned[i]
        adx_val = adx_aligned[i]
        bb_middle_val = bb_middle_aligned[i]
        bb_upper_val = bb_upper_aligned[i]
        bb_lower_val = bb_lower_aligned[i]
        
        # Volume filter: current volume > 1.3 * 20-period average
        vol_spike = vol > 1.3 * vol_ma
        
        if position == 0:
            if adx_val > 25:  # Trending market - follow trend with EMA
                # Long conditions: price above EMA + volume spike
                if price > ema_val and vol_spike:
                    signals[i] = 0.25
                    position = 1
                # Short conditions: price below EMA + volume spike
                elif price < ema_val and vol_spike:
                    signals[i] = -0.25
                    position = -1
            else:  # Ranging market - mean reversion at Bollinger Bands
                # Long conditions: price at or below lower BB + volume spike
                if price <= bb_lower_val and vol_spike:
                    signals[i] = 0.25
                    position = 1
                # Short conditions: price at or above upper BB + volume spike
                elif price >= bb_upper_val and vol_spike:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                if adx_val > 25:  # trending - exit when price crosses below EMA
                    if price < ema_val:
                        exit_signal = True
                else:  # ranging - exit when price reaches middle BB
                    if price >= bb_middle_val:
                        exit_signal = True
            
            elif position == -1:  # short position
                if adx_val > 25:  # trending - exit when price crosses above EMA
                    if price > ema_val:
                        exit_signal = True
                else:  # ranging - exit when price reaches middle BB
                    if price <= bb_middle_val:
                        exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Regime_ADX_BB_EMA_Volume"
timeframe = "6h"
leverage = 1.0