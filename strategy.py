#!/usr/bin/env python3
# 6h_RSI_Divergence_1dTrend_VolumeFilter
# Hypothesis: RSI divergence (bullish/bearish) on 6h chart combined with 1-day EMA trend filter and volume confirmation.
# Bullish divergence: price makes lower low while RSI makes higher low -> potential reversal up.
# Bearish divergence: price makes higher high while RSI makes lower high -> potential reversal down.
# Works in both bull and bear markets by catching reversals at extremes.
# Volume filter ensures institutional participation. 1-day trend filter avoids counter-trend trades.
# Targets 15-30 trades per year on 6h timeframe to minimize fee drag.

name = "6h_RSI_Divergence_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily EMA for trend filter (34-period)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate RSI (14-period) on 6h close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Find local minima and maxima for divergence detection
    # For bullish divergence: look for price lower low with RSI higher low
    # For bearish divergence: look for price higher high with RSI lower high
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need RSI (14), EMA (34), and enough data for divergence detection
    start_idx = 50  # Sufficient warmup for RSI and divergence detection
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(rsi_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation (20-period MA on 6h chart)
        if i >= 20:
            volume_ma = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > volume_ma * 1.5
        else:
            volume_confirm = False
        
        # RSI divergence detection (look back 5 periods for swing points)
        bullish_div = False
        bearish_div = False
        
        if i >= 5:
            # Check for bullish divergence: price lower low, RSI higher low
            if low[i] < low[i-5] and rsi_values[i] > rsi_values[i-5]:
                # Additional confirmation: RSI should be in oversold territory (< 40)
                if rsi_values[i] < 40:
                    bullish_div = True
            
            # Check for bearish divergence: price higher high, RSI lower high
            if high[i] > high[i-5] and rsi_values[i] < rsi_values[i-5]:
                # Additional confirmation: RSI should be in overbought territory (> 60)
                if rsi_values[i] > 60:
                    bearish_div = True
        
        if position == 0:
            # Long entry: bullish divergence + daily uptrend + volume confirmation
            if bullish_div and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish divergence + daily downtrend + volume confirmation
            elif bearish_div and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish divergence or trend turns down or RSI overbought
            if bearish_div or not uptrend or rsi_values[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish divergence or trend turns up or RSI oversold
            if bullish_div or not downtrend or rsi_values[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals