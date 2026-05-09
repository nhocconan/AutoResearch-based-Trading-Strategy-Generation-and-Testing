#!/usr/bin/env python3
# 4h_RSI_Div_With_Volume_Trend
# Strategy: RSI divergence with volume confirmation and 1d trend filter
# Long: Bullish RSI divergence + price > 1d EMA50 + volume spike
# Short: Bearish RSI divergence + price < 1d EMA50 + volume spike
# Exit: RSI crosses opposite extreme (long exit at RSI>70, short exit at RSI<30)
# Designed for 4h timeframe with selective entries to minimize trade frequency and work in bull/bear markets

name = "4h_RSI_Div_With_Volume_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[1:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    gain_smooth = wilders_smooth(gain, 14)
    loss_smooth = wilders_smooth(loss, 14)
    
    rs = np.where(loss_smooth != 0, gain_smooth / loss_smooth, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate volume spike (volume > 1.5 * 20-period average)
    vol_ma = np.zeros_like(volume)
    vol_sum = 0.0
    vol_count = 0
    for i in range(len(volume)):
        vol_sum += volume[i]
        vol_count += 1
        if vol_count > 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        if vol_count >= 20:
            vol_ma[i] = vol_sum / 20.0
        else:
            vol_ma[i] = 0.0
    
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 0.0)
    vol_spike = vol_ratio > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema_50_aligned[i]) or i < 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check for RSI divergence
        bullish_div = False
        bearish_div = False
        
        # Look back 5 periods for divergence
        lookback = 5
        if i >= lookback:
            # Bullish divergence: price makes lower low, RSI makes higher low
            if close[i] < close[i-lookback] and rsi[i] > rsi[i-lookback]:
                # Check if it's a meaningful low
                if rsi[i] < 40 and rsi[i-lookback] < 40:
                    bullish_div = True
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            if close[i] > close[i-lookback] and rsi[i] < rsi[i-lookback]:
                # Check if it's a meaningful high
                if rsi[i] > 60 and rsi[i-lookback] > 60:
                    bearish_div = True
        
        if position == 0:
            # Enter long: Bullish RSI divergence + above 1d EMA50 + volume spike
            if bullish_div and close[i] > ema_50_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Bearish RSI divergence + below 1d EMA50 + volume spike
            elif bearish_div and close[i] < ema_50_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI crosses above 70 (overbought)
            if rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI crosses below 30 (oversold)
            if rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals