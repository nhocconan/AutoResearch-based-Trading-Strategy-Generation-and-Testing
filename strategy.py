#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Index with daily ADX trend filter and weekly volatility filter
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Long when Bull Power > 0 and Bear Power < 0 (bullish momentum) + daily ADX > 25 (trending) + weekly ATR ratio < 0.8 (low vol)
# Short when Bull Power < 0 and Bear Power > 0 (bearish momentum) + daily ADX > 25 + weekly ATR ratio < 0.8
# Exit when Elder Ray signals reverse or volatility expands (weekly ATR ratio > 1.2)
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25
# Uses daily ADX for trend strength and weekly ATR regime filter to avoid choppy markets
# Target: 75-150 total trades over 4 years (19-38/year)

name = "6h_elder_ray_1d_adx_1w_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1-day data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1-week data for ATR regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 13-period EMA for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).values
    
    # Elder Ray components
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 1-day ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0]) * -1  # invert to positive
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / (tr_14 + 1e-10)
    minus_di = 100 * minus_dm_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1-week ATR (14-period) for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1_w = high_1w - low_1w
    tr2_w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_w = np.abs(low_1w - np.roll(close_1w, 1))
    tr2_w[0] = tr1_w[0]
    tr3_w[0] = tr1_w[0]
    tr_1w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    
    atr_1w = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # ATR(14) for stoploss (using primary timeframe)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(atr_1w_aligned[i]) or 
            np.isnan(atr[i])):
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
            # Exit: Elder Ray turns bearish OR volatility expands
            elif (bull_power[i] <= 0 or bear_power[i] >= 0) or atr_1w_aligned[i] > 1.2 * atr_1w[i-30]:  # 30-period lookback for ATR reference
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
            # Exit: Elder Ray turns bullish OR volatility expands
            elif (bull_power[i] >= 0 or bear_power[i] <= 0) or atr_1w_aligned[i] > 1.2 * atr_1w[i-30]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Elder Ray signals with ADX trend filter and low volatility regime
            # Trend filter: daily ADX > 25
            trend_filter = adx_1d_aligned[i] > 25
            # Volatility filter: weekly ATR < 80% of its 30-period average (low vol regime)
            atr_ma_30 = pd.Series(atr_1w_aligned).rolling(window=30, min_periods=30).mean().values
            vol_filter = atr_1w_aligned[i] < 0.8 * atr_ma_30[i]
            
            # Long: Bullish Elder Ray (Bull Power > 0 and Bear Power < 0) + trend + low vol
            if bull_power[i] > 0 and bear_power[i] < 0 and trend_filter and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Bearish Elder Ray (Bull Power < 0 and Bear Power > 0) + trend + low vol
            elif bull_power[i] < 0 and bear_power[i] > 0 and trend_filter and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals