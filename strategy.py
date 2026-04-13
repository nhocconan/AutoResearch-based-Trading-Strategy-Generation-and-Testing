#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h trend + 1h mean reversion in ranging markets.
    # Long when 4h EMA20 uptrend, ADX < 20 (range), and price touches 1h VWAP with rejection.
    # Short when 4h EMA20 downtrend, ADX < 20, and price rejects 1h VWAP.
    # Uses 4h for trend filter, 1h for entry timing to reduce whipsaw.
    # Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA20 for trend
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Calculate 1h ADX(14) for regime
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1h VWAP
    typical_price = (high + low + close) / 3
    pv = typical_price * volume
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(volume)
    vwap = cum_pv / cum_vol
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(adx[i]) or np.isnan(vwap[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 4h EMA20 slope
        if i >= 51:
            ema20_prev = ema20_4h_aligned[i-1]
            ema20_curr = ema20_4h_aligned[i]
            trend_up = ema20_curr > ema20_prev
            trend_down = ema20_curr < ema20_prev
        else:
            trend_up = False
            trend_down = False
        
        # Regime filter: only trade in ranging markets (ADX < 20)
        ranging = adx[i] < 20
        
        # VWAP rejection conditions (price touches VWAP and closes back inside range)
        long_setup = (low[i] <= vwap[i]) and (close[i] > vwap[i]) and trend_up and ranging
        short_setup = (high[i] >= vwap[i]) and (close[i] < vwap[i]) and trend_down and ranging
        
        # Exit conditions: price returns to 4h EMA20
        long_exit = close[i] >= ema20_4h_aligned[i]
        short_exit = close[i] <= ema20_4h_aligned[i]
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.20
        
        # Entry conditions
        if long_setup and position != 1:
            position = 1
            signals[i] = position_size
        elif short_setup and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1h_ema20_vwap_rejection_v1"
timeframe = "1h"
leverage = 1.0