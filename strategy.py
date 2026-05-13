#!/usr/bin/env python3
# Hypothesis: 6h Bollinger Band Squeeze Breakout with 12h Trend Filter and Volume Confirmation.
# Uses Bollinger Band Width percentile to detect low volatility squeeze (regime filter).
# Breakout occurs when price closes outside Bollinger Bands after squeeze.
# Trend filter: 12h EMA50 direction (price > EMA50 = bullish, price < EMA50 = bearish).
# Volume confirmation: current 6h volume > 1.5x 20-period average.
# Works in bull via breakout continuation, bear via faded rallies (mean reversion in squeeze).
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_BollingerSqueeze_Breakout_12hTrend_Volume_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 6h data for Bollinger Bands
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # Bollinger Bands (20, 2) on 6h close
    bb_period = 20
    bb_std = 2
    ma_6h = pd.Series(close_6h).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_6h = pd.Series(close_6h).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = ma_6h + (bb_std * std_6h)
    lower_bb = ma_6h - (bb_std * std_6h)
    
    # Bollinger Band Width for squeeze detection
    bb_width = (upper_bb - lower_bb) / ma_6h
    # Percentile of BB width over 50 periods to identify squeeze (low volatility)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    squeeze_condition = bb_width_percentile < 20  # Bottom 20% = squeeze
    
    # Breakout condition: price closes outside Bollinger Bands
    breakout_up = close_6h > upper_bb
    breakout_down = close_6h < lower_bb
    
    # Volume filter: current 6h volume > 1.5x 20-period average
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_filter_6h = volume_6h > (1.5 * vol_ma_6h)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA(50) on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(ma_6h[i]) or np.isnan(std_6h[i]) or
            np.isnan(bb_width_percentile[i]) or np.isnan(ema50_12h_aligned[i]) or
            np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Breakout above upper BB during squeeze AND bullish trend (price > 12h EMA50) AND volume confirmation
            if squeeze_condition[i] and breakout_up[i] and close[i] > ema50_12h_aligned[i] and volume_filter_6h[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakout below lower BB during squeeze AND bearish trend (price < 12h EMA50) AND volume confirmation
            elif squeeze_condition[i] and breakout_down[i] and close[i] < ema50_12h_aligned[i] and volume_filter_6h[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters Bollinger Bands (mean reversion) OR trend reversal
            if close[i] < ma_6h[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters Bollinger Bands (mean reversion) OR trend reversal
            if close[i] > ma_6h[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals