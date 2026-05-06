#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h EMA(21) trend with 1d RSI(14) mean reversion for entries
# Uses 4h EMA for trend direction and 1d RSI for oversold/overbought entries
# Entry long when price > EMA21 AND RSI < 30 (oversold in uptrend)
# Entry short when price < EMA21 AND RSI > 70 (overbought in downtrend)
# Exit when RSI reverts to neutral (40-60 range) or opposite signal
# Volume confirmation (>1.5x 20-bar average) ensures participation
# Designed for both bull/bear markets: trend filter avoids counter-trend traps,
# RSI mean reversion captures bounces in trends and reversals in extremes
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag

name = "4h_EMA21_1dRSI_MeanRev_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 4h EMA21 trend filter
    close_series = pd.Series(close)
    ema21 = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 1d RSI(14)
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    
    # Calculate volume filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align HTF indicators to 4h timeframe (primary)
    ema21_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), ema21)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema21_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price > EMA21 (uptrend) AND RSI < 30 (oversold) AND volume
            if close[i] > ema21_aligned[i] and rsi_1d_aligned[i] < 30 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price < EMA21 (downtrend) AND RSI > 70 (overbought) AND volume
            elif close[i] < ema21_aligned[i] and rsi_1d_aligned[i] > 70 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI reverts to neutral (>=40) or opposite signal
            if rsi_1d_aligned[i] >= 40 or close[i] < ema21_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI reverts to neutral (<=60) or opposite signal
            if rsi_1d_aligned[i] <= 60 or close[i] > ema21_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals