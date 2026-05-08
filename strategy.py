#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour ADX trend filter with 1-week RSI mean reversion and volume confirmation
# Long when weekly RSI < 30 (oversold) + weekly close above weekly SMA(50) + daily ADX > 25 (trending)
# Short when weekly RSI > 70 (overbought) + weekly close below weekly SMA(50) + daily ADX > 25 (trending)
# Weekly RSI identifies extremes; weekly SMA(50) filters false signals; daily ADX confirms trend strength
# Volume spike confirms institutional participation at reversal points
# Targets 50-150 total trades over 4 years (12-37/year) to avoid fee drag

name = "12h_ADX_WeeklyRSI_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data once for RSI and SMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly RSI(14)
    weekly_close = df_1w['close'].values
    delta = np.diff(weekly_close, prepend=weekly_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_aligned = align_htf_to_ltf(prices, df_1w, rsi_14)
    
    # Calculate weekly SMA(50) for trend filter
    sma50_1w = pd.Series(weekly_close).rolling(window=50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    # Get daily data once for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily ADX(14)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(daily_high[1:] - daily_low[1:])
    tr2 = np.abs(daily_high[1:] - daily_close_1d[:-1])
    tr3 = np.abs(daily_low[1:] - daily_close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = np.concatenate([[np.nan], daily_high[1:] - daily_high[:-1]])
    down_move = np.concatenate([[np.nan], daily_low[:-1] - daily_low[1:]])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / tr_14
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / tr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_14 = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_14_aligned[i]) or np.isnan(sma50_1w_aligned[i]) or 
            np.isnan(adx_14_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi_val = rsi_14_aligned[i]
        sma50_val = sma50_1w_aligned[i]
        adx_val = adx_14_aligned[i]
        vol_spike = volume_spike[i]
        weekly_close_val = df_1w['close'].values[np.searchsorted(df_1w.index, prices.index[i])] if i >= len(df_1w) else df_1w['close'].values[min(i, len(df_1w)-1)]
        # Simplified: use the aligned SMA directly
        weekly_close_aligned = close  # proxy, but we'll use price vs SMA50
        
        if position == 0:
            # Enter long: RSI < 30 (oversold) + weekly close above SMA(50) + ADX > 25 + volume spike
            if rsi_val < 30 and close[i] > sma50_val and adx_val > 25 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: RSI > 70 (overbought) + weekly close below SMA(50) + ADX > 25 + volume spike
            elif rsi_val > 70 and close[i] < sma50_val and adx_val > 25 and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI > 50 (neutral) OR ADX < 20 (weak trend)
            if rsi_val > 50 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI < 50 (neutral) OR ADX < 20 (weak trend)
            if rsi_val < 50 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals