#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-week RSI divergence with volume confirmation and Bollinger Bands
# Long when weekly RSI < 30 and price > 12h Bollinger middle band with volume > 1.5x 20-period average
# Short when weekly RSI > 70 and price < 12h Bollinger middle band with volume > 1.5x 20-period average
# Uses weekly RSI for overbought/oversold conditions, Bollinger Bands for mean reversion entry
# Designed to work in bull markets via mean reversion from oversold and in bear markets via mean reversion from overbought
# Target: 20-30 trades per year (80-120 over 4 years) with 0.25 position sizing

name = "12h_1wRSI_Bollinger_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-week RSI (14-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # 14-period RSI calculation
    delta = pd.Series(df_1w['close']).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align weekly RSI to 12h timeframe
    rsi_12h = align_htf_to_ltf(prices, df_1w, rsi_values)
    
    # Calculate 12h Bollinger Bands (20-period, 2 std dev)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    middle_bb = sma_20  # 20-period SMA
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Bollinger warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(rsi_12h[i]) or np.isnan(middle_bb[i]) or 
            np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: RSI oversold (<30) and price above middle Bollinger Band with volume confirmation
            if rsi_12h[i] < 30 and close[i] > middle_bb[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: RSI overbought (>70) and price below middle Bollinger Band with volume confirmation
            elif rsi_12h[i] > 70 and close[i] < middle_bb[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI crosses above 50 (momentum shift) or price touches upper Bollinger Band
            if rsi_12h[i] > 50 or close[i] >= upper_bb[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI crosses below 50 (momentum shift) or price touches lower Bollinger Band
            if rsi_12h[i] < 50 or close[i] <= lower_bb[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals