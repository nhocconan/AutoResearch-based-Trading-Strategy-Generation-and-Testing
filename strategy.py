#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray with 1d regime filter.
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Bullish when Bull Power > 0 and rising; Bearish when Bear Power < 0 and falling.
# Regime filter: 1d ADX > 25 for trending, < 20 for ranging.
# In trending regime (ADX > 25): follow Elder Ray signals.
# In ranging regime (ADX < 20): fade extreme Elder Ray (bull power > 0.5*ATR or bear power < -0.5*ATR).
# Uses ATR-based stoploss. Designed to work in both bull and bear markets by adapting to regime.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_elderray_1d_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    tr_smooth = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # EMA13 on 6h for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = high - ema13
    bear_power = low - ema13
    
    # ATR for volatility normalization and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(atr[i]) or np.isnan(adx_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        adx_val = adx_1d_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        atr_val = atr[i]
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR below entry
            if close[i] < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit conditions based on regime
            elif adx_val > 25:  # trending regime
                # Exit when bear power becomes positive (momentum shifts)
                if bear_val > 0:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                else:
                    signals[i] = 0.25
            else:  # ranging regime (ADX < 25)
                # Exit when bull power normalizes
                if bull_val < 0.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                else:
                    signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR above entry
            if close[i] > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit conditions based on regime
            elif adx_val > 25:  # trending regime
                # Exit when bull power becomes negative (momentum shifts)
                if bull_val < 0:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                else:
                    signals[i] = -0.25
            else:  # ranging regime (ADX < 25)
                # Exit when bear power normalizes
                if bear_val > -0.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                else:
                    signals[i] = -0.25
        else:
            # Look for entries based on regime
            if adx_val > 25:  # trending regime - follow Elder Ray
                # Long: bull power positive AND rising (bullish momentum)
                if i > 30 and bull_val > 0 and bull_val > bull_power[i-1]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: bear power negative AND falling (bearish momentum)
                elif i > 30 and bear_val < 0 and bear_val < bear_power[i-1]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:  # ranging regime (ADX < 25) - fade extremes
                # Long: bear power extremely negative (oversold)
                if bear_val < -0.5 * atr_val:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: bull power extremely positive (overbought)
                elif bull_val > 0.5 * atr_val:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals