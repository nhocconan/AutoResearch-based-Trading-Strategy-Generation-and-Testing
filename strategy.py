#!/usr/bin/env python3
"""
4h_1d_1w_Triple_Timeframe_Trend_Following
Hypothesis: Combines trend alignment across 4h, 1d, and 1w timeframes with volume confirmation and volatility filtering.
In strong trends (price above/below key EMAs on all three timeframes), enters long when 4h closes above 20 EMA with volume > 1.5x average,
and short when 4h closes below 20 EMA with volume confirmation. Uses 1w ADX to filter for trending markets only.
Designed to capture major trends while avoiding choppy markets, working in both bull and bear cycles by following the dominant trend.
Target: 20-50 trades/year on 4h (80-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMAs (20 and 50)
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ADX (14)
    period = 14
    tr1 = pd.Series(high_1w).rolling(2).max() - pd.Series(low_1w).rolling(2).min()
    tr2 = abs(pd.Series(high_1w).rolling(2).max() - pd.Series(close_1w).shift(1))
    tr3 = abs(pd.Series(low_1w).rolling(2).min() - pd.Series(close_1w).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    plus_dm = pd.Series(high_1w).diff()
    minus_dm = pd.Series(low_1w).diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr_ma = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di_ma = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / tr_ma)
    minus_di_ma = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / tr_ma)
    dx = (abs(plus_di_ma - minus_di_ma) / (plus_di_ma + minus_di_ma)) * 100
    adx_1w = dx.ewm(alpha=1/period, adjust=False).mean().values
    
    # Get 4h data for entry signals
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h EMA (20) for entry
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 4h volume average (20)
    vol_ma_20_4h = pd.Series(volume_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if any data is not ready
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(adx_1w_aligned[i]) or np.isnan(ema_20_4h_aligned[i]) or
            np.isnan(vol_ma_20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend alignment: price above/both EMAs on 1d
        bullish_alignment = close_1d[i] > ema_20_1d_aligned[i] and close_1d[i] > ema_50_1d_aligned[i]
        bearish_alignment = close_1d[i] < ema_20_1d_aligned[i] and close_1d[i] < ema_50_1d_aligned[i]
        
        # Strong trend filter: 1w ADX > 25
        strong_trend = adx_1w_aligned[i] > 25
        
        # Volume confirmation: 4h volume > 1.5x 20 EMA volume
        volume_confirmation = volume_4h[i] > (vol_ma_20_4h_aligned[i] * 1.5)
        
        # Entry conditions
        if bullish_alignment and strong_trend and volume_confirmation:
            # Long when 4h price crosses above 20 EMA
            if close_4h[i] > ema_20_4h_aligned[i] and (i == 50 or close_4h[i-1] <= ema_20_4h_aligned[i-1]):
                if position != 1:
                    position = 1
                    signals[i] = position_size
                else:
                    signals[i] = position_size
            # Hold long position
            elif position == 1:
                signals[i] = position_size
            else:
                signals[i] = 0.0
        elif bearish_alignment and strong_trend and volume_confirmation:
            # Short when 4h price crosses below 20 EMA
            if close_4h[i] < ema_20_4h_aligned[i] and (i == 50 or close_4h[i-1] >= ema_20_4h_aligned[i-1]):
                if position != -1:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = -position_size
            # Hold short position
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        else:
            # Exit conditions: trend breaks or low ADX
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_1w_Triple_Timeframe_Trend_Following"
timeframe = "4h"
leverage = 1.0