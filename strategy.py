#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using weekly RSI mean reversion with 1-day volume filter and Bollinger Bands for confirmation
# Long when weekly RSI < 30 and price is below daily Bollinger Lower Band with volume > 1.5x average
# Short when weekly RSI > 70 and price is above daily Bollinger Upper Band with volume > 1.5x average
# Weekly RSI provides oversold/overbought signals on higher timeframe, Bollinger Bands provide entry/exit timing on daily,
# Volume confirms momentum shift. Works in both bull and bear markets by capturing mean reversion extremes.
# Target: 12-37 trades per year (50-150 over 4 years) with 0.25 position sizing.

name = "12h_weeklyRSI_BollingerVolume_MeanRev_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly RSI (14-period) ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # RSI calculation (14-period)
    delta = pd.Series(df_1w['close']).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align weekly RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi_values)
    
    # Calculate daily Bollinger Bands (20-period, 2 std)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Bollinger Bands calculation
    sma_20 = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + (2 * std_20)
    bb_lower = sma_20 - (2 * std_20)
    
    # Align Bollinger Bands to 12h timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    
    # Volume confirmation: >1.5x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (1.5 * vol_ma_30)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(rsi_aligned[i]) or np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long signal: weekly RSI oversold (<30) and price below lower Bollinger Band with volume confirmation
            if rsi_aligned[i] < 30 and close[i] < bb_lower_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short signal: weekly RSI overbought (>70) and price above upper Bollinger Band with volume confirmation
            elif rsi_aligned[i] > 70 and close[i] > bb_upper_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: weekly RSI returns to neutral (>50) or price reaches middle Bollinger Band
            if rsi_aligned[i] > 50 or close[i] > (bb_upper_aligned[i] + bb_lower_aligned[i]) / 2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: weekly RSI returns to neutral (<50) or price reaches middle Bollinger Band
            if rsi_aligned[i] < 50 or close[i] < (bb_upper_aligned[i] + bb_lower_aligned[i]) / 2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals