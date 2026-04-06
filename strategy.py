#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h ADX trend strength filter + 4h RSI mean reversion + session filter.
# Long when ADX > 25 (trending) and 4h RSI < 30 (oversold) during 08-20 UTC.
# Short when ADX > 25 and 4h RSI > 70 (overbought) during 08-20 UTC.
# Uses ADX to avoid ranging markets and RSI for mean reversion entries.
# Session filter reduces noise outside active hours. Target: 60-150 total trades over 4 years.

name = "1h_adx25_4hrsi_meanrev_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # ADX calculation (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # 4h RSI (14-period)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # RSI calculation
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if ADX or RSI data not available
        if np.isnan(adx[i]) or np.isnan(rsi_4h_aligned[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: ADX < 20 (trend weakening) or RSI reverts to midpoint
        if position == 1:  # long position
            if (adx[i] < 20 or 
                rsi_4h_aligned[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if (adx[i] < 20 or 
                rsi_4h_aligned[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with ADX > 25 (strong trend) and session filter
            if adx[i] > 25 and in_session[i]:
                # Long: 4h RSI < 30 (oversold)
                if rsi_4h_aligned[i] < 30:
                    signals[i] = 0.20
                    position = 1
                # Short: 4h RSI > 70 (overbought)
                elif rsi_4h_aligned[i] > 70:
                    signals[i] = -0.20
                    position = -1
    
    return signals