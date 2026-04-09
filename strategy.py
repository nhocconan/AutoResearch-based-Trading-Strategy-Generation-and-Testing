#!/usr/bin/env python3
# 1h_macd_rsi_4h1d_trend_v2
# Hypothesis: 1h strategy using MACD histogram cross + RSI pullback for entry timing,
# with 4h EMA trend filter and 1d ADX regime filter. Designed for low trade frequency
# (target: 60-150 total trades over 4 years) to avoid fee drag. Works in bull/bear
# by using trend filter (4h EMA) and regime filter (1d ADX < 25 = range, > 25 = trend).
# Uses discrete sizing (±0.20) to minimize fee churn.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_macd_rsi_4h1d_trend_v2"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC) - open_time is already datetime64[ms]
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h HTF data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d HTF data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
    # True Range
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d) - pd.Series(high_1d).shift(1)
    down_move = pd.Series(low_1d).shift(1) - pd.Series(low_1d)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / (atr_1d + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr_1d + 1e-10)
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1h indicators for entry timing
    # MACD (12,26,9)
    ema_fast = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_slow = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    
    # RSI (14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(macd_hist[i]) or np.isnan(signal_line[i]) or
            np.isnan(rsi[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 4h EMA
        trend_up = close[i] > ema_4h_aligned[i]
        trend_down = close[i] < ema_4h_aligned[i]
        
        # Determine regime from 1d ADX
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] <= 25
        
        if position == 1:  # Long position
            # Exit: trend reversal OR MACD histogram turns negative
            if not trend_up or macd_hist[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: trend reversal OR MACD histogram turns positive
            if not trend_down or macd_hist[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Volume confirmation: current volume > 1.5x 20-period average
            volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
            volume_confirmed = volume[i] > 1.5 * volume_ma[i] if not np.isnan(volume_ma[i]) else False
            
            if volume_confirmed:
                # Long conditions: uptrend + MACD bullish cross + RSI pullback (not overbought)
                if trend_up and macd_hist[i] > 0 and macd_hist[i-1] <= 0 and rsi[i] < 70:
                    position = 1
                    signals[i] = 0.20
                # Short conditions: downtrend + MACD bearish cross + RSI pullback (not oversold)
                elif trend_down and macd_hist[i] < 0 and macd_hist[i-1] >= 0 and rsi[i] > 30:
                    position = -1
                    signals[i] = -0.20
    
    return signals