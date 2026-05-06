#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h RSI extremes with volume confirmation and trend filter
# RSI < 30 on 12h indicates oversold with potential bounce, RSI > 70 indicates overbought with potential pullback
# Volume > 2x 20-period average confirms momentum behind the move
# Trend filter: 50-period EMA on 4h to align with higher timeframe momentum
# Works in bull/bear markets: oversold bounces in bear markets, overbought pullbacks in bull markets
# Target: 20-50 total trades over 4 years (5-12/year) with 0.30 position sizing to minimize fee drag

name = "4h_RSI12_Extreme_VolumeTrendFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h RSI ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # 14-period RSI on 12h close
    close_12h = pd.Series(df_12h['close'])
    delta = close_12h.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h_values = rsi_12h.values
    
    # Align 12h RSI to 4h timeframe
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h_values)
    
    # Volume confirmation: >2.0x 20-period average (high threshold to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_20)
    
    # Trend filter: 50-period EMA on 4h
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend = close > ema_50
    downtrend = close < ema_50
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(rsi_12h_aligned[i]) or np.isnan(volume_filter[i]) or np.isnan(ema_50[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) with volume confirmation and uptrend
            if rsi_12h_aligned[i] < 30 and volume_filter[i] and uptrend[i]:
                signals[i] = 0.30
                position = 1
            # Short: RSI > 70 (overbought) with volume confirmation and downtrend
            elif rsi_12h_aligned[i] > 70 and volume_filter[i] and downtrend[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: RSI returns to neutral (50) or reverses to overbought
            if rsi_12h_aligned[i] >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: RSI returns to neutral (50) or reverses to oversold
            if rsi_12h_aligned[i] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals