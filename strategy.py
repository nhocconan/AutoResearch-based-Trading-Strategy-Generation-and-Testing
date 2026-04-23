#!/usr/bin/env python3
"""
Hypothesis: 1-day Bollinger Band squeeze with 1-week trend filter and volume confirmation.
Long when price breaks above upper BB during low volatility (BB width < 20th percentile) and weekly uptrend.
Short when price breaks below lower BB during low volatility and weekly downtrend.
Uses Bollinger Bands to identify low-volatility breakouts, which often precede strong moves in both bull and bear markets.
Weekly trend filter ensures we only trade in the direction of the higher timeframe trend.
Target: 10-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for Bollinger Bands - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Load 1-week data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1-day Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    
    # Calculate BB width percentile (20-period lookback for regime)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Calculate 1-week EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = close_1w > ema_50_1w
    weekly_downtrend = close_1w < ema_50_1w
    
    # Align HTF indicators to lower timeframe
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # Volume average (20-period) on lower timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(bb_width_percentile_aligned[i]) or np.isnan(weekly_uptrend_aligned[i]) or
            np.isnan(weekly_downtrend_aligned[i]) or np.isnan(upper_bb_aligned[i]) or
            np.isnan(lower_bb_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bb_width_pct = bb_width_percentile_aligned[i]
        is_uptrend = weekly_uptrend_aligned[i] > 0.5
        is_downtrend = weekly_downtrend_aligned[i] > 0.5
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price_close = close[i]
        upper_bb_val = upper_bb_aligned[i]
        lower_bb_val = lower_bb_aligned[i]
        
        if position == 0:
            # Long: BB squeeze breakout above upper BB, weekly uptrend, volume confirmation
            if (bb_width_pct < 20 and  # Low volatility (squeeze)
                price_close > upper_bb_val and
                is_uptrend and
                vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: BB squeeze breakout below lower BB, weekly downtrend, volume confirmation
            elif (bb_width_pct < 20 and  # Low volatility (squeeze)
                  price_close < lower_bb_val and
                  is_downtrend and
                  vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: volatility expansion or mean reversion
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to middle of BB or volatility expands
                if (price_close < sma_20[-1] if len(sma_20) > 0 else False) or bb_width_pct > 50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to middle of BB or volatility expands
                if (price_close > sma_20[-1] if len(sma_20) > 0 else False) or bb_width_pct > 50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_BollingerSqueeze_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0