#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w RSI momentum with volume confirmation and Bollinger Bands mean reversion.
# In bull markets: buy when weekly RSI > 50 and price touches lower BB with volume spike.
# In bear markets: sell when weekly RSI < 50 and price touches upper BB with volume spike.
# Uses weekly trend filter to avoid counter-trend trades. Designed for low frequency (10-20 trades/year).

name = "1d_WeeklyRSI_BB_Volume_MeanReversion"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for RSI trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly RSI(14)
    delta = np.diff(close_1w)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1w)
    avg_loss = np.zeros_like(close_1w)
    
    # Wilder's smoothing
    avg_gain[13] = np.mean(gain[:14])
    avg_loss[13] = np.mean(loss[:14])
    
    for i in range(14, len(close_1w)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w[:14] = np.nan
    
    # Align weekly RSI to daily
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Daily Bollinger Bands(20, 2)
    close_series = pd.Series(close)
    sma_20 = close_series.rolling(window=20, min_periods=20).mean()
    std_20 = close_series.rolling(window=20, min_periods=20).std()
    bb_upper = (sma_20 + 2 * std_20).values
    bb_lower = (sma_20 - 2 * std_20).values
    
    # Volume confirmation: volume > 2x 20-day average
    vol_ma = close_series.rolling(window=20, min_periods=20).mean()
    vol_ma_array = vol_ma.values
    vol_confirm = volume > (vol_ma_array * 2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for BB
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(vol_ma_array[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: weekly bullish bias + price at lower BB + volume spike
            if rsi_1w_aligned[i] > 50 and close[i] <= bb_lower[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short setup: weekly bearish bias + price at upper BB + volume spike
            elif rsi_1w_aligned[i] < 50 and close[i] >= bb_upper[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches middle BB or weekly RSI turns bearish
            if close[i] >= sma_20.iloc[i] or rsi_1w_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches middle BB or weekly RSI turns bullish
            if close[i] <= sma_20.iloc[i] or rsi_1w_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals