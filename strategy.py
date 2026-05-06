#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-day RSI with Bollinger Bands mean reversion
# In bear/range markets (2025+), price tends to revert to mean after extreme RSI moves
# RSI(14) > 70 with price touching upper Bollinger Band (20,2) = short signal
# RSI(14) < 30 with price touching lower Bollinger Band (20,2) = long signal
# Volume confirmation: current volume > 1.5x 20-period average to avoid false signals
# Trend filter: 50-period EMA on 6x timeframe to avoid counter-trend trades in strong trends
# Works in bull/bear: mean reversion works in ranges, trend filter avoids whipsaws in trends
# Target: 60-120 total trades over 4 years (15-30/year) with 0.25 position sizing

name = "6h_RSI_Bollinger_MeanReversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily RSI and Bollinger Bands ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:  # Need enough for RSI(14) and BB(20)
        return np.zeros(n)
    
    # Daily RSI(14)
    close_1d = pd.Series(df_1d['close'].values)
    delta = close_1d.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = (100 - (100 / (1 + rs))).values
    
    # Daily Bollinger Bands (20,2)
    sma_20 = close_1d.rolling(window=20, min_periods=20).mean()
    std_20 = close_1d.rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    
    # Align daily indicators to 6h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb.values)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb.values)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Trend filter: 50-period EMA on 6h timeframe
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend = close > ema_50
    downtrend = close < ema_50
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(rsi_aligned[i]) or np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or
            np.isnan(volume_filter[i]) or np.isnan(ema_50[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long signal: RSI < 30 (oversold) and price at or below lower Bollinger Band
            if rsi_aligned[i] < 30 and close[i] <= lower_bb_aligned[i] and volume_filter[i] and uptrend[i]:
                signals[i] = 0.25
                position = 1
            # Short signal: RSI > 70 (overbought) and price at or above upper Bollinger Band
            elif rsi_aligned[i] > 70 and close[i] >= upper_bb_aligned[i] and volume_filter[i] and downtrend[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI returns to neutral (50) or price reaches middle of Bollinger Bands
            bb_middle = (upper_bb_aligned[i] + lower_bb_aligned[i]) / 2
            if rsi_aligned[i] >= 50 or close[i] >= bb_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI returns to neutral (50) or price reaches middle of Bollinger Bands
            bb_middle = (upper_bb_aligned[i] + lower_bb_aligned[i]) / 2
            if rsi_aligned[i] <= 50 or close[i] <= bb_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals