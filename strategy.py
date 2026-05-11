#!/usr/bin/env python3
"""
1d_1w_Engulfing_Signal_TrendFilter
Hypothesis: Daily bullish/bearish engulfing candles at weekly support/resistance, filtered by weekly trend (ADX > 25), produce high-probability swing trades. Works in bull/bear markets by only trading with the weekly trend. Volume confirmation avoids false signals. Targets 20-50 trades over 4 years on 1d timeframe.
"""

name = "1d_1w_Engulfing_Signal_TrendFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend and support/resistance
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily OHLCV
    open_1d = prices['open'].values
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    volume_1d = prices['volume'].values
    
    # --- Weekly Trend: ADX(14) ---
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_w - low_w)
    tr2 = np.abs(high_w - np.roll(close_w, 1))
    tr3 = np.abs(low_w - np.roll(close_w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.diff(high_w, prepend=high_w[0])
    down_move = -np.diff(low_w, prepend=low_w[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_w
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_w
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_w = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_w_aligned = align_htf_to_ltf(prices, df_1w, adx_w)
    
    # --- Weekly Support/Resistance: Previous Week High/Low ---
    prev_week_high = np.roll(df_1w['high'].values, 1)
    prev_week_low = np.roll(df_1w['low'].values, 1)
    prev_week_high[0] = df_1w['high'].values[0]
    prev_week_low[0] = df_1w['low'].values[0]
    
    weekly_high = align_htf_to_ltf(prices, df_1w, prev_week_high)
    weekly_low = align_htf_to_ltf(prices, df_1w, prev_week_low)
    
    # --- Daily Volume Average for confirmation ---
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup
    start_idx = 35  # for ADX and volume average
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(adx_w_aligned[i]) or np.isnan(weekly_high[i]) or 
            np.isnan(weekly_low[i]) or np.isnan(vol_avg_1d[i])):
            if position != 0:
                # Simple stoploss: 2.5x ATR estimate from entry
                atr_est = np.abs(high_1d[i] - low_1d[i])
                if position == 1 and close_1d[i] <= entry_price - 2.5 * atr_est:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_1d[i] >= entry_price + 2.5 * atr_est:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine if weekly trend is strong enough
        strong_trend = adx_w_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.3x daily average
        vol_confirm = volume_1d[i] > 1.3 * vol_avg_1d[i]
        
        # Detect daily engulfing candles
        bullish_engulf = (close_1d[i] > open_1d[i-1]) and (open_1d[i] < close_1d[i-1])
        bearish_engulf = (open_1d[i] > close_1d[i-1]) and (close_1d[i] < open_1d[i-1])
        
        if position == 0:
            # Look for entries only with strong weekly trend and volume confirmation
            if strong_trend and vol_confirm:
                # Bullish engulfing near weekly support
                if bullish_engulf and low_1d[i] <= weekly_low[i] * 1.001:  # within 0.1% of weekly low
                    signals[i] = 0.25
                    position = 1
                    entry_price = close_1d[i]
                # Bearish engulfing near weekly resistance
                elif bearish_engulf and high_1d[i] >= weekly_high[i] * 0.999:  # within 0.1% of weekly high
                    signals[i] = -0.25
                    position = -1
                    entry_price = close_1d[i]
        else:
            # Manage existing position
            if position == 1:
                # Long position: exit on bearish engulfing or stoploss
                if bearish_engulf:
                    signals[i] = 0.0
                    position = 0
                # Stoploss: 2.5x ATR below entry
                elif close_1d[i] <= entry_price - 2.5 * np.abs(high_1d[i] - low_1d[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short position: exit on bullish engulfing or stoploss
                if bullish_engulf:
                    signals[i] = 0.0
                    position = 0
                # Stoploss: 2.5x ATR above entry
                elif close_1d[i] >= entry_price + 2.5 * np.abs(high_1d[i] - low_1d[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals