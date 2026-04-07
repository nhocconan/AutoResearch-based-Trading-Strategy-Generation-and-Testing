#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Camarilla Pivot + 1d Trend + Volume Confirmation
# Hypothesis: Camarilla pivot levels from daily timeframe identify key support/resistance zones.
# In ranging markets (ADX < 25), we mean-revert at inner levels (L3/L4, H3/H4).
# In trending markets (ADX > 25), we breakout on breaches of outer levels (L4/H4) with trend confirmation.
# Volume confirms institutional participation. This adapts to both ranging and trending markets.
# 4h timeframe balances responsiveness and noise reduction. Target: 20-50 trades/year (80-200 over 4 years).
name = "4h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from daily data
    # Formulas: H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    #           H3 = close + 1.125*(high-low), L3 = close - 1.125*(high-low)
    #           H2 = close + 0.75*(high-low), L2 = close - 0.75*(high-low)
    #           H1 = close + 0.375*(high-low), L1 = close - 0.375*(high-low)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    daily_range = daily_high - daily_low
    
    # Camarilla levels
    H4 = daily_close + 1.5 * daily_range
    L4 = daily_close - 1.5 * daily_range
    H3 = daily_close + 1.125 * daily_range
    L3 = daily_close - 1.125 * daily_range
    H2 = daily_close + 0.75 * daily_range
    L2 = daily_close - 0.75 * daily_range
    H1 = daily_close + 0.375 * daily_range
    L1 = daily_close - 0.375 * daily_range
    
    # Align Camarilla levels to 4h timeframe
    H4_4h = align_htf_to_ltf(prices, df_1d, H4)
    L4_4h = align_htf_to_ltf(prices, df_1d, L4)
    H3_4h = align_htf_to_ltf(prices, df_1d, H3)
    L3_4h = align_htf_to_ltf(prices, df_1d, L3)
    H2_4h = align_htf_to_ltf(prices, df_1d, H2)
    L2_4h = align_htf_to_ltf(prices, df_1d, L2)
    H1_4h = align_htf_to_ltf(prices, df_1d, H1)
    L1_4h = align_htf_to_ltf(prices, df_1d, L1)
    
    # 1-day EMA(50) for trend filter
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    daily_ema_4h = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # ADX(14) for regime detection on 4h timeframe
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum.reduce([tr1, tr2, tr3])])
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    # Smoothed values
    tr14 = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    plus_dm14 = pd.Series(plus_dm).ewm(span=14, adjust=False).mean().values
    minus_dm14 = pd.Series(minus_dm).ewm(span=14, adjust=False).mean().values
    # Directional Indicators
    plus_di = 100 * plus_dm14 / tr14
    minus_di = 100 * minus_dm14 / tr14
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(H4_4h[i]) or np.isnan(L4_4h[i]) or np.isnan(H3_4h[i]) or np.isnan(L3_4h[i]) or
            np.isnan(daily_ema_4h[i]) or np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches L3 (mean reversion) or breaks below L4 with volume in trend
            if close[i] <= L3_4h[i] or (close[i] < L4_4h[i] and vol_filter[i] and adx[i] > 25):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price reaches H3 (mean reversion) or breaks above H4 with volume in trend
            if close[i] >= H3_4h[i] or (close[i] > H4_4h[i] and vol_filter[i] and adx[i] > 25):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                if adx[i] < 25:  # Ranging market: mean reversion at inner levels
                    # Long: price touches/bounces off L3
                    if abs(close[i] - L3_4h[i]) < 0.001 * L3_4h[i]:
                        position = 1
                        signals[i] = 0.25
                    # Short: price touches/bounces off H3
                    elif abs(close[i] - H3_4h[i]) < 0.001 * H3_4h[i]:
                        position = -1
                        signals[i] = -0.25
                else:  # Trending market: breakout of outer levels with trend confirmation
                    # Long: price breaks above H4 with trend confirmation
                    if close[i] > H4_4h[i] and close[i] > daily_ema_4h[i]:
                        position = 1
                        signals[i] = 0.25
                    # Short: price breaks below L4 with trend confirmation
                    elif close[i] < L4_4h[i] and close[i] < daily_ema_4h[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals