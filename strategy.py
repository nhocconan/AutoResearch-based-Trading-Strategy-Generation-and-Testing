#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion with 4h Bollinger Squeeze + 1d trend filter.
# Uses Bollinger Band width contraction (squeeze) on 4h as volatility regime filter.
# Enters long when price < lower BB + RSI < 30, short when price > upper BB + RSI > 70.
# 1d EMA50 determines long-only/short-only bias to avoid counter-trend trades.
# Session filter (08-20 UTC) reduces noise. Target: 15-30 trades/year.

name = "1h_BB_Squeeze_RSI_4hVol_1dTrend"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Bollinger Bands
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Bollinger Bands (20, 2) on 4h
    bb_period = 20
    bb_std = 2
    
    sma_20 = np.full(len(close_4h), np.nan)
    std_20 = np.full(len(close_4h), np.nan)
    for i in range(bb_period, len(close_4h)):
        sma_20[i] = np.mean(close_4h[i-bb_period:i])
        std_20[i] = np.std(close_4h[i-bb_period:i])
    
    upper_bb = sma_20 + bb_std * std_20
    lower_bb = sma_20 - bb_std * std_20
    bb_width = (upper_bb - lower_bb) / sma_20  # normalized width
    
    # Bollinger Squeeze: width < 20th percentile of last 50 bars
    squeeze = np.full(len(bb_width), False)
    for i in range(50, len(bb_width)):
        if not np.isnan(bb_width[i]):
            historical_widths = bb_width[i-50:i]
            valid_widths = historical_widths[~np.isnan(historical_widths)]
            if len(valid_widths) >= 10:
                p20 = np.percentile(valid_widths, 20)
                squeeze[i] = bb_width[i] < p20
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # EMA50 on 1d
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 + ema_50_1d[i-1] * 49) / 50
    
    # Trend: close > EMA50 = uptrend bias (long only), close < EMA50 = downtrend bias (short only)
    uptrend_bias = np.full(len(close_1d), False)
    downtrend_bias = np.full(len(close_1d), False)
    for i in range(len(close_1d)):
        if not np.isnan(ema_50_1d[i]):
            uptrend_bias[i] = close_1d[i] > ema_50_1d[i]
            downtrend_bias[i] = close_1d[i] < ema_50_1d[i]
    
    # RSI(14) on 1h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    rsi = np.full(n, np.nan)
    
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
        else:
            rsi[i] = 100
    
    # Align all indicators to 1h timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_4h, squeeze)
    uptrend_bias_aligned = align_htf_to_ltf(prices, df_1d, uptrend_bias)
    downtrend_bias_aligned = align_htf_to_ltf(prices, df_1d, downtrend_bias)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 50, 14)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(rsi[i]) or np.isnan(squeeze_aligned[i]) or np.isnan(uptrend_bias_aligned[i]) or np.isnan(downtrend_bias_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for entry: BB squeeze + RSI extreme + trend bias
            # Long when RSI < 30 in uptrend bias + squeeze
            long_condition = (rsi[i] < 30) and squeeze_aligned[i] and uptrend_bias_aligned[i]
            # Short when RSI > 70 in downtrend bias + squeeze
            short_condition = (rsi[i] > 70) and squeeze_aligned[i] and downtrend_bias_aligned[i]
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: RSI > 50 or no longer in uptrend bias
            if (rsi[i] > 50) or not uptrend_bias_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: RSI < 50 or no longer in downtrend bias
            if (rsi[i] < 50) or not downtrend_bias_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals