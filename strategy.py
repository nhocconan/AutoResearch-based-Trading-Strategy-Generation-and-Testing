#!/usr/bin/env python3
# 4h_Engulfing_Breakout_Trend_Momentum
# Hypothesis: Combines bullish/bearish engulfing patterns with 4h RSI momentum and 1d EMA trend filter.
# Engulfing patterns signal strong momentum shifts; RSI confirms overbought/oversold conditions are not extreme;
# EMA filter ensures trades align with higher-timeframe trend. Works in bull markets by catching continuation
# of uptrends after pullbacks, and in bear markets by catching continuation of downtrends after bounces.
# Volume confirmation filters low-conviction signals. Designed for moderate trade frequency (~20-40/year).

name = "4h_Engulfing_Breakout_Trend_Momentum"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_prices = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h RSI (14) for momentum filter
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily EMA34 (34), RSI (14), volume MA (20)
    start_idx = max(34, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(rsi_values[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # RSI momentum filter (not overbought/oversold)
        rsi_not_extreme = (rsi_values[i] > 30) and (rsi_values[i] < 70)
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Bullish engulfing: current green candle fully engulfs previous red candle
        bullish_engulfing = (close[i] > open_prices[i]) and \
                           (open_prices[i-1] > close[i-1]) and \
                           (close[i] >= open_prices[i-1]) and \
                           (open_prices[i] <= close[i-1])
        
        # Bearish engulfing: current red candle fully engulfs previous green candle
        bearish_engulfing = (close[i] < open_prices[i]) and \
                           (open_prices[i-1] < close[i-1]) and \
                           (close[i] <= open_prices[i-1]) and \
                           (open_prices[i] >= close[i-1])
        
        if position == 0:
            # Long entry: uptrend + bullish engulfing + RSI not extreme + volume
            if uptrend and bullish_engulfing and rsi_not_extreme and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + bearish engulfing + RSI not extreme + volume
            elif downtrend and bearish_engulfing and rsi_not_extreme and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or bearish engulfing forms
            if not uptrend or bearish_engulfing:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or bullish engulfing forms
            if not downtrend or bullish_engulfing:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals