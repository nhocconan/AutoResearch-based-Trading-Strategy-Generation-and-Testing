#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Choppiness_Reversal_Signal_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Choppiness Index (trend/range filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1d
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[0.0], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate ATR(14) for 1d
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ADX(14) components for 1d
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smooth DM and TR for ADX
    atr_period = 14
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(alpha=1/atr_period, adjust=False).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(alpha=1/atr_period, adjust=False).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = pd.Series(dx_1d).ewm(alpha=1/atr_period, adjust=False).mean().values
    
    # Calculate Choppiness Index (14-period)
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_1d = 100 * np.log10(sum_tr_14 / (highest_high_14 - lowest_low_14 + 1e-10)) / np.log10(14)
    
    # Align HTF indicators to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # 12h RSI(14) for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_12h = 100 - (100 / (1 + rs))
    
    # Volume spike filter: current volume > 1.8x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(rsi_12h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: Only trade in ranging markets (Choppiness > 61.8)
        # and avoid strong trends (ADX < 25)
        ranging_market = chop_1d_aligned[i] > 61.8
        weak_trend = adx_1d_aligned[i] < 25
        
        if position == 0:
            # Mean reversion signals in ranging market
            if ranging_market and weak_trend:
                # Oversold RSI + volume spike = long
                if rsi_12h[i] < 30 and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Overbought RSI + volume spike = short
                elif rsi_12h[i] > 70 and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: RSI returns to neutral or overbought
            if rsi_12h[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI returns to neutral or oversold
            if rsi_12h[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals