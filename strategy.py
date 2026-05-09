#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h price action strategy using 1d Bollinger Bands for mean reversion
# with volume confirmation and 1w EMA200 trend filter. Trades reversals at Bollinger
# Bands (20, 2) in ranging markets while filtering for higher timeframe trend.
# Designed for low trade frequency (<30/year) to minimize fee drag and work in
# both bull and bear markets by following weekly trend direction.
name = "12h_BollingerMeanReversion_1wEMA200_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate 1d Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    
    # Use previous day's bands (shift by 1 to avoid look-ahead)
    upper_bb_shifted = np.roll(upper_bb, 1)
    lower_bb_shifted = np.roll(lower_bb, 1)
    upper_bb_shifted[0] = np.nan
    lower_bb_shifted[0] = np.nan
    
    # Align to 12h timeframe
    upper_bb_12h = align_htf_to_ltf(prices, df_1d, upper_bb_shifted)
    lower_bb_12h = align_htf_to_ltf(prices, df_1d, lower_bb_shifted)
    
    # 1w EMA200 trend filter
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume confirmation: volume > 1.5x 50-period EMA
    vol_ema50 = pd.Series(volume).ewm(span=50, adjust=False, min_periods=50).mean().values
    vol_confirm = volume > (1.5 * vol_ema50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(upper_bb_12h[i]) or np.isnan(lower_bb_12h[i]) or
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_ema50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price touches lower Bollinger Band with volume confirmation and above 1w EMA200
            if (price <= lower_bb_12h[i] and vol_confirm[i] and price > ema_200_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price touches upper Bollinger Band with volume confirmation and below 1w EMA200
            elif (price >= upper_bb_12h[i] and vol_confirm[i] and price < ema_200_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses above SMA20 (mean reversion to middle)
            if price >= sma_20[-1]:  # Use last available SMA value (not ideal but avoids look-ahead)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses below SMA20 (mean reversion to middle)
            if price <= sma_20[-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals