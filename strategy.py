#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and session filter
# - Long when price breaks above H3 pivot level AND 4h EMA(20) > EMA(50) (bullish trend) AND hour in [08,20] UTC
# - Short when price breaks below L3 pivot level AND 4h EMA(20) < EMA(50) (bearish trend) AND hour in [08,20] UTC
# - Exit when price returns to Pivot Point (mean reversion to equilibrium)
# - Uses discrete position sizing (0.20) to minimize fee churn
# - Camarilla pivots provide precise intraday support/resistance levels
# - 4h EMA filter ensures alignment with higher timeframe trend
# - Session filter (08-20 UTC) reduces noise during low-liquidity hours
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
# - Works in both bull and bear markets: breakouts in trends, mean reversion in ranges

name = "1h_4h_camarilla_breakout_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Pre-compute 4h EMA trend filter: EMA(20) vs EMA(50)
    close_4h = df_4h['close'].values
    ema_20 = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_50 = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_bullish = ema_20 > ema_50
    ema_bearish = ema_20 < ema_50
    
    # Align 4h EMA trend to 1h timeframe
    ema_bullish_aligned = align_htf_to_ltf(prices, df_4h, ema_bullish)
    ema_bearish_aligned = align_htf_to_ltf(prices, df_4h, ema_bearish)
    
    # Pre-compute daily Camarilla pivot levels (using prior day's OHLC)
    # Need daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivots for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: Pivot = (H+L+C)/3
    # H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    h3_1d = close_1d + (high_1d - low_1d) * 1.1 / 4.0
    l3_1d = close_1d - (high_1d - low_1d) * 1.1 / 4.0
    
    # Align daily Camarilla levels to 1h timeframe (with 1-bar delay for completed day)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_bullish_aligned[i]) or np.isnan(ema_bearish_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(in_session[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H3 AND 4h bullish trend AND in session
            if (prices['close'].iloc[i] > h3_aligned[i] and 
                ema_bullish_aligned[i] and 
                in_session[i]):
                position = 1
                signals[i] = 0.20
            # Short when price breaks below L3 AND 4h bearish trend AND in session
            elif (prices['close'].iloc[i] < l3_aligned[i] and 
                  ema_bearish_aligned[i] and 
                  in_session[i]):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to Pivot Point (mean reversion)
            # Exit when price returns to Pivot Point (within 0.1% tolerance)
            exit_signal = np.abs(prices['close'].iloc[i] - pivot_aligned[i]) / pivot_aligned[i] < 0.001
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
    
    return signals