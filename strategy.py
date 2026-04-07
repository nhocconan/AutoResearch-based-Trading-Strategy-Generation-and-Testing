#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Index with 1-day trend filter and Bollinger Bands mean reversion
# Elder Ray = Bull Power (High - EMA13) and Bear Power (EMA13 - Low)
# Long when Bull Power > 0 and Bear Power < 0 (bullish momentum) + price below BB lower band (oversold)
# Short when Bear Power > 0 and Bull Power < 0 (bearish momentum) + price above BB upper band (overbought)
# Uses 1-day EMA trend filter: only long when price > daily EMA50, only short when price < daily EMA50
# Exit when Elder Ray signals reverse or price crosses EMA13
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Combines momentum (Elder Ray) with mean reversion (BB) and trend filter (daily EMA)
# Target: 75-200 total trades over 4 years (19-50/year)

name = "6h_elder_ray_bb_1d_ema_v1"
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
    
    # 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Calculate Bollinger Bands (20, 2.0)
    bb_mid = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 1-day EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Elder Ray turns bearish OR price crosses below EMA13
            elif bull_power[i] <= 0 or bear_power[i] >= 0 or close[i] < ema13[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Elder Ray turns bullish OR price crosses above EMA13
            elif bear_power[i] <= 0 or bull_power[i] >= 0 or close[i] > ema13[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Elder Ray signals with BB mean reversion and daily EMA trend filter
            # Elder Ray bullish: Bull Power > 0 and Bear Power < 0
            elder_bullish = bull_power[i] > 0 and bear_power[i] < 0
            # Elder Ray bearish: Bear Power > 0 and Bull Power < 0
            elder_bearish = bear_power[i] > 0 and bull_power[i] < 0
            # BB oversold: price below lower band
            bb_oversold = close[i] < bb_lower[i]
            # BB overbought: price above upper band
            bb_overbought = close[i] > bb_upper[i]
            # Trend filter: price > daily EMA50 for long, price < daily EMA50 for short
            uptrend = close[i] > ema50_1d_aligned[i]
            downtrend = close[i] < ema50_1d_aligned[i]
            
            # Long: Elder Ray bullish + BB oversold + uptrend
            if elder_bullish and bb_oversold and uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Elder Ray bearish + BB overbought + downtrend
            elif elder_bearish and bb_overbought and downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals