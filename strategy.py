#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h EMA200 for trend direction and 1h RSI with volume confirmation for entry.
# 4h EMA200 > 1h close = bullish bias, < 1h close = bearish bias to trade with higher timeframe trend.
# RSI(14) < 30 for long, > 70 for short with volume > 1.5x 20-period average to avoid false signals.
# Designed to work in both bull and bear markets by using 4h trend filter to avoid counter-trend trades.
# Target: 60-150 total trades over 4 years = 15-37/year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE for EMA200 calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate EMA200 on 4h data
    close_4h = df_4h['close'].values
    ema200 = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_aligned = align_htf_to_ltf(prices, df_4h, ema200)
    
    # Calculate RSI on 1h data
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    rsi_period = 14
    avg_gain = pd.Series(gain).ewm(span=rsi_period, adjust=False, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, adjust=False, min_periods=rsi_period).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = max(200, 20)  # Need EMA200 and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema200_aligned[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: 4h EMA200 direction
        bullish_trend = close[i] > ema200_aligned[i]
        bearish_trend = close[i] < ema200_aligned[i]
        
        if position == 0:
            # Look for RSI extremes with volume confirmation
            # Long: RSI < 30 (oversold) AND bullish trend on 4h
            if (rsi[i] < 30 and 
                bullish_trend and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: RSI > 70 (overbought) AND bearish trend on 4h
            elif (rsi[i] > 70 and 
                  bearish_trend and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral (50) or trend changes
            if (rsi[i] >= 50 or 
                not bullish_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI returns to neutral (50) or trend changes
            if (rsi[i] <= 50 or 
                not bearish_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4hEMA200_RSI_VolumeFilter_v1"
timeframe = "1h"
leverage = 1.0