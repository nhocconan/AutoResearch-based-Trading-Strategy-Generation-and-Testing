#!/usr/bin/env python3
# 6h_RSI_Extremes_With_Volume_Regime
# Hypothesis: RSI extremes (overbought/oversold) on 6h timeframe combined with volume confirmation and ADX regime filter identifies mean-reversion opportunities in both trending and ranging markets. Volume confirms momentum behind the move, while ADX avoids choppy conditions where mean reversion fails. Designed for low frequency (15-30 trades/year) to minimize fee drag.

name = "6h_RSI_Extremes_With_Volume_Regime"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 6h RSI (14-period)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Daily ADX (14-period) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(high_1d).diff()
    tr3 = pd.Series(close_1d).diff()
    tr1 = pd.Series(high_1d) - pd.Series(low_1d)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = pd.Series(low_1d).diff()
    up_move = up_move.where(up_move > down_move, 0)
    down_move = (-down_move).where(down_move > up_move, 0)
    
    plus_di = 100 * (up_move.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr)
    minus_di = 100 * (down_move.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_values = adx.values
    
    # Align daily ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Volume confirmation: 20-period average
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi_values[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        # Regime filter: only trade when ADX < 25 (ranging market)
        ranging_market = adx_aligned[i] < 25
        
        if position == 0:
            # Enter long: RSI oversold with volume confirmation in ranging market
            if (rsi_values[i] < 30 and volume_confirm and ranging_market):
                signals[i] = 0.25
                position = 1
            # Enter short: RSI overbought with volume confirmation in ranging market
            elif (rsi_values[i] > 70 and volume_confirm and ranging_market):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when RSI returns to neutral or volume fails
            if (rsi_values[i] > 50 or not volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when RSI returns to neutral or volume fails
            if (rsi_values[i] < 50 or not volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals