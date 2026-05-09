#!/usr/bin/env python3
# Hypothesis: 4h timeframe with daily Bollinger Band squeeze and RSI mean reversion.
# Uses daily Bollinger Band width percentile to identify low volatility (squeeze) conditions.
# In squeeze, enters long when RSI < 30 and price > 200 EMA, short when RSI > 70 and price < 200 EMA.
# Exits when Bollinger Band width expands beyond 50th percentile or RSI reverts to 50.
# Designed to work in both bull and bear markets by exploiting mean reversion during low volatility.
# Target: 75-200 total trades over 4 years (19-50/year) with size 0.25.

name = "4h_BollingerSqueeze_RSI_MeanReversion"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for Bollinger Bands and EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    
    # Calculate Bollinger Band width percentile (252-day lookback for 1 year)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Calculate daily EMA200 for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate daily RSI (14)
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    
    # Align daily indicators to 4h timeframe
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Squeeze condition: BB width below 20th percentile (low volatility)
    squeeze_condition = bb_width_percentile_aligned < 20
    
    # Mean reversion conditions
    rsi_oversold = rsi_1d_aligned < 30
    rsi_overbought = rsi_1d_aligned > 70
    price_above_ema = close > ema_200_1d_aligned
    price_below_ema = close < ema_200_1d_aligned
    
    # Exit conditions: BB width expands above 50th percentile or RSI returns to 50
    expand_condition = bb_width_percentile_aligned > 50
    rsi_neutral = (rsi_1d_aligned > 45) & (rsi_1d_aligned < 55)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(squeeze_condition[i]) or np.isnan(rsi_oversold[i]) or
            np.isnan(rsi_overbought[i]) or np.isnan(price_above_ema[i]) or
            np.isnan(price_below_ema[i]) or np.isnan(expand_condition[i]) or
            np.isnan(rsi_neutral[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: squeeze + RSI oversold + price above EMA200
            if squeeze_condition[i] and rsi_oversold[i] and price_above_ema[i]:
                signals[i] = 0.25
                position = 1
            # Short: squeeze + RSI overbought + price below EMA200
            elif squeeze_condition[i] and rsi_overbought[i] and price_below_ema[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: volatility expansion or RSI returns to neutral
            if expand_condition[i] or rsi_neutral[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: volatility expansion or RSI returns to neutral
            if expand_condition[i] or rsi_neutral[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals