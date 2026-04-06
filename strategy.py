#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h ADX trend filter and 1d volatility filter.
# Long when price breaks above 1h EMA(20) during strong ADX trend (ADX>25) and low volatility day.
# Short when price breaks below 1h EMA(20) during strong ADX trend and low volatility day.
# Uses session filter (08-20 UTC) to avoid low-liquidity hours.
# Target: 80-160 total trades over 4 years (20-40/year) for optimal balance.

name = "1h_ema20_adx_vol_session_v1"
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
    volume = prices['volume'].values
    
    # 1h EMA(20)
    close_s = pd.Series(close)
    ema20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 4h ADX(14) for trend strength
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = high_4h[1:] - high_4h[:-1]
    down_move = low_4h[:-1] - low_4h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI and DX
    plus_di = 100 * plus_dm14 / tr14
    minus_di = 100 * minus_dm14 / tr14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # 1d volatility filter: low volatility day (ATR ratio < 1.2)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1d = np.maximum.reduce([
        high_1d[1:] - low_1d[1:],
        np.abs(high_1d[1:] - close_1d[:-1]),
        np.abs(low_1d[1:] - close_1d[:-1])
    ])
    tr1d = np.concatenate([[np.nan], tr1d])
    atr1d = pd.Series(tr1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr1d_ma = pd.Series(atr1d).rolling(window=10, min_periods=10).mean().values
    low_vol_day = atr1d < (atr1d_ma * 1.2)
    
    # Align HTF indicators
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    low_vol_aligned = align_htf_to_ltf(prices, df_1d, low_vol_day)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
            
        # Skip if ADX data not available or volatility data not available
        if np.isnan(adx_aligned[i]) or np.isnan(low_vol_aligned[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Strong trend and low volatility conditions
        strong_trend = adx_aligned[i] > 25
        low_volatility = low_vol_aligned[i]
        
        # Check exits
        if position == 1:  # long position
            # Exit: price closes below EMA20 or trend weakens
            if (close[i] < ema20[i]) or (not strong_trend) or (not low_volatility):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price closes above EMA20 or trend weakens
            if (close[i] > ema20[i]) or (not strong_trend) or (not low_volatility):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with trend and volatility filters
            if strong_trend and low_volatility:
                # Long: price crosses above EMA20
                if (close[i] > ema20[i]) and (close[i-1] <= ema20[i-1]):
                    signals[i] = 0.20
                    position = 1
                # Short: price crosses below EMA20
                elif (close[i] < ema20[i]) and (close[i-1] >= ema20[i-1]):
                    signals[i] = -0.20
                    position = -1
    
    return signals