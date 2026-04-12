#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d_KAMA_RSI_Chop_v1
# Uses 1-day KAMA to determine trend direction (bullish/bearish) and RSI for mean reversion entries.
# In trending markets (KAMA slope > 0): buy dips when RSI < 30, sell when RSI > 70.
# In ranging markets (KAMA slope <= 0): mean revert at RSI extremes with tighter bounds (RSI < 25 or > 75).
# Uses weekly ADX as regime filter: only trade when ADX > 25 (trending) or ADX < 20 (ranging).
# Weekly volatility filter: only trade when weekly ATR ratio (current/10-period avg) < 1.5 to avoid chaotic markets.
# Designed for low trade frequency (target: 20-50 trades/year) with clear regime adaptation.
# Works in both bull (follow KAMA trend) and bear (mean reversion in ranges) markets.

name = "1d_KAMA_RSI_Chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for ADX and ATR filters
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly ADX (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed DM and ATR
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Weekly ATR for volatility filter (current ATR / 10-period average)
    atr_10_avg = pd.Series(atr_1w).rolling(window=10, min_periods=10).mean().values
    atr_ratio = atr_1w / atr_10_avg
    vol_filter = atr_ratio < 1.5  # Avoid high volatility periods
    
    # Align weekly filters to daily
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    vol_filter_aligned = align_htf_to_ltf(prices, df_1w, vol_filter)
    
    # Get daily data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    # Efficiency Ratio
    change = np.abs(np.concatenate([[np.nan], close_1d[1:] - close_1d[:-1]]))
    volatility = pd.Series(np.abs(np.concatenate([[np.nan], close_1d[1:] - close_1d[:-1]]))).rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # Start after 10 periods
    for i in range(10, len(close_1d)):
        if np.isnan(kama[i-1]) or np.isnan(sc[i]):
            kama[i] = close_1d[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # KAMA slope for trend direction
    kama_slope = np.concatenate([[np.nan], kama[1:] - kama[:-1]])
    
    # Calculate RSI (14-period)
    delta = np.concatenate([[np.nan], close_1d[1:] - close_1d[:-1]])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Align daily indicators
    kama_slope_aligned = align_htf_to_ltf(prices, df_1d, kama_slope)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # start after warmup
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(vol_filter_aligned[i]) or 
            np.isnan(kama_slope_aligned[i]) or np.isnan(rsi_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filters: ADX > 25 (trending) or ADX < 20 (ranging) AND low volatility
        if not ((adx_aligned[i] > 25 or adx_aligned[i] < 20) and vol_filter_aligned[i]):
            # Hold current position if filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine market regime
        is_trending = adx_aligned[i] > 25
        
        if is_trending:
            # Trending market: follow KAMA direction
            if kama_slope_aligned[i] > 0 and rsi_aligned[i] < 30 and position != 1:
                # Buy dip in uptrend
                position = 1
                signals[i] = 0.25
            elif kama_slope_aligned[i] < 0 and rsi_aligned[i] > 70 and position != -1:
                # Sell rally in downtrend
                position = -1
                signals[i] = -0.25
            elif position == 1 and rsi_aligned[i] > 70:
                # Exit long on overbought
                position = 0
                signals[i] = 0.0
            elif position == -1 and rsi_aligned[i] < 30:
                # Exit short on oversold
                position = 0
                signals[i] = 0.0
            else:
                # Hold position
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        else:
            # Ranging market: mean reversion at RSI extremes
            if rsi_aligned[i] < 25 and position != 1:
                # Buy oversold
                position = 1
                signals[i] = 0.25
            elif rsi_aligned[i] > 75 and position != -1:
                # Sell overbought
                position = -1
                signals[i] = -0.25
            elif position == 1 and rsi_aligned[i] > 60:
                # Exit long at moderate RSI
                position = 0
                signals[i] = 0.0
            elif position == -1 and rsi_aligned[i] < 40:
                # Exit short at moderate RSI
                position = 0
                signals[i] = 0.0
            else:
                # Hold position
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
    
    return signals