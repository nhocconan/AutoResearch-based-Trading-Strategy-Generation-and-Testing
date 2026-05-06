#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Bollinger Bands with RSI mean reversion and volume confirmation.
# Long when price touches or crosses below weekly Bollinger Lower Band with RSI(14) < 30 and volume > 1.5x 20-period average.
# Short when price touches or crosses above weekly Bollinger Upper Band with RSI(14) > 70 and volume > 1.5x 20-period average.
# Exit when price crosses back to weekly Bollinger Middle Band (20-period SMA).
# Uses weekly timeframe for structure and mean reversion, daily for execution.
# Designed to work in both bull and bear markets via mean reversion at volatility extremes.
# Target: 15-25 trades per year (60-100 over 4 years) with 0.25 position sizing.

name = "1d_weeklyBB_RSI_Volume_MR_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly Bollinger Bands (20-period SMA, 2 std dev)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    sma_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    middle_bb = sma_20
    
    # Align weekly BB to daily timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1w, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1w, lower_bb)
    middle_bb_aligned = align_htf_to_ltf(prices, df_1w, middle_bb)
    
    # RSI(14) on daily timeframe
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(middle_bb_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price at or below lower BB, RSI oversold, volume confirmation
            if close[i] <= lower_bb_aligned[i] and rsi[i] < 30 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price at or above upper BB, RSI overbought, volume confirmation
            elif close[i] >= upper_bb_aligned[i] and rsi[i] > 70 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back to middle BB
            if close[i] >= middle_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back to middle BB
            if close[i] <= middle_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals