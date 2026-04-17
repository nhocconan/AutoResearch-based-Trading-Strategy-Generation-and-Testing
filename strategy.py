#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly 3-week high/low breakout with volume confirmation and volatility filter
# Weekly highs/lows act as strong support/resistance; breakouts with volume indicate institutional interest
# Works in both bull/bear markets: breakouts capture new trends, volatility filter avoids chop
# Target: 1d timeframe with 1h trend filter to reduce false signals
# Expected trades: 20-40/year (80-160 total over 4 years) to stay within limits

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for 3-week high/low calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 3-week high and low (using last 3 weekly bars)
    # We need at least 3 weeks of data
    if len(high_1w) < 3:
        return np.zeros(n)
    
    # Calculate rolling 3-week high and low
    high_3w = pd.Series(high_1w).rolling(window=3, min_periods=3).max().values
    low_3w = pd.Series(low_1w).rolling(window=3, min_periods=3).min().values
    
    # Shift to use previous 3-week high/low (avoid look-ahead)
    high_3w_prev = np.roll(high_3w, 1)
    low_3w_prev = np.roll(low_3w, 1)
    high_3w_prev[0] = np.nan
    low_3w_prev[0] = np.nan
    
    # Align weekly 3-week high/low to daily timeframe
    high_3w_daily = align_htf_to_ltf(prices, df_1w, high_3w_prev)
    low_3w_daily = align_htf_to_ltf(prices, df_1w, low_3w_prev)
    
    # Get 1h trend filter (use 4h as proxy for trend since 1h not available in MTF list)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # Calculate 20-period EMA on 4h close
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    # Align 4h EMA to daily timeframe
    ema_20_4h_daily = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Volume confirmation: current volume > 1.5 * 20-day average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volatility filter: ATR(14) > ATR(50) to avoid low volatility environments
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_slow = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma20[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(atr_slow[i]) or 
            np.isnan(high_3w_daily[i]) or 
            np.isnan(low_3w_daily[i]) or
            np.isnan(ema_20_4h_daily[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-day average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        # Volatility filter: ATR(14) > ATR(50) (avoid low volatility)
        volatility_filter = atr[i] > atr_slow[i]
        # Trend filter: price above/below 4h EMA20
        uptrend = close[i] > ema_20_4h_daily[i]
        downtrend = close[i] < ema_20_4h_daily[i]
        
        if position == 0:
            # Long: price breaks above 3-week high with volume, volatility, and uptrend
            if close[i] > high_3w_daily[i] and volume_filter and volatility_filter and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 3-week low with volume, volatility, and downtrend
            elif close[i] < low_3w_daily[i] and volume_filter and volatility_filter and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below 3-week low or trend turns down
            if close[i] < low_3w_daily[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above 3-week high or trend turns up
            if close[i] > high_3w_daily[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_3Week_High_Low_Breakout_Vol_Trend"
timeframe = "1d"
leverage = 1.0