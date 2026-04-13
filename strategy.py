#!/usr/bin/env python3
"""
4h_1d_1w_Triple_Timeframe_Confluence
Hypothesis: Combines 1w trend filter, 1d momentum, and 4h entry timing to capture medium-term moves in both bull and bear markets.
Goes long when 1w EMA21 > EMA50 (bullish trend), 1d RSI > 50 (bullish momentum), and 4h close > 4h EMA21 with volume > 1.5x average.
Goes short when 1w EMA21 < EMA50 (bearish trend), 1d RSI < 50 (bearish momentum), and 4h close < 4h EMA21 with volume > 1.5x average.
Uses discrete position sizing (0.25) to minimize fee churn and includes ATR-based stoploss.
Target: 20-50 trades per year on 4h (80-200 total over 4 years).
"""

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA21 and EMA50 for trend filter
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get daily data for momentum
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily RSI(14)
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    
    # Get 4h data for entry timing
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h EMA21 for entry
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate volume average for confirmation
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR for stoploss
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_4h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align all signals to 4h timeframe
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    atr_multiplier = 2.5  # ATR multiplier for stoploss
    
    # Track entry price for stoploss
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema21_1w_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or
            np.isnan(rsi_1d_aligned[i]) or np.isnan(ema21_4h_aligned[i]) or
            np.isnan(vol_ma_20_4h_aligned[i]) or np.isnan(atr_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend and momentum
        bullish_trend = ema21_1w_aligned[i] > ema50_1w_aligned[i]
        bearish_trend = ema21_1w_aligned[i] < ema50_1w_aligned[i]
        bullish_momentum = rsi_1d_aligned[i] > 50
        bearish_momentum = rsi_1d_aligned[i] < 50
        
        # Volume confirmation
        volume_confirmation = volume[i] > (vol_ma_20_4h_aligned[i] * 1.5)
        
        # Update entry price and position side when position changes
        if position == 1 and position_side[i-1] != 1:
            entry_price[i] = close[i]
            position_side[i] = 1
        elif position == -1 and position_side[i-1] != -1:
            entry_price[i] = close[i]
            position_side[i] = -1
        else:
            # Carry forward
            entry_price[i] = entry_price[i-1] if i > 0 else 0
            position_side[i] = position_side[i-1] if i > 0 else 0
        
        # Check stoploss
        stoploss_triggered = False
        if position == 1 and i > 0:
            if close[i] < entry_price[i] - (atr_multiplier * atr_4h_aligned[i]):
                stoploss_triggered = True
        elif position == -1 and i > 0:
            if close[i] > entry_price[i] + (atr_multiplier * atr_4h_aligned[i]):
                stoploss_triggered = True
        
        if stoploss_triggered:
            position = 0
            position_side[i] = 0
            signals[i] = 0.0
            continue
        
        # Entry logic
        if bullish_trend and bullish_momentum and volume_confirmation:
            # Long conditions
            if close[i] > ema21_4h_aligned[i]:
                if position != 1:
                    position = 1
                    signals[i] = position_size
                else:
                    signals[i] = position_size
            else:
                # Hold or exit
                if position == 1:
                    signals[i] = position_size
                else:
                    signals[i] = 0.0
                    position = 0
        elif bearish_trend and bearish_momentum and volume_confirmation:
            # Short conditions
            if close[i] < ema21_4h_aligned[i]:
                if position != -1:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = -position_size
            else:
                # Hold or exit
                if position == -1:
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
                    position = 0
        else:
            # No clear signal - exit
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_1w_Triple_Timeframe_Confluence"
timeframe = "4h"
leverage = 1.0