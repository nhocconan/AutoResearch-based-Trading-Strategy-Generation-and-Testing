#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour RSI mean reversion with 1-day trend filter and volume confirmation.
# RSI(14) on 4h identifies overbought (>70) and oversold (<30) conditions for mean reversion.
# 1-day EMA(50) provides trend direction - only trade counter-trend when RSI extreme aligns with trend exhaustion.
# Volume confirmation ensures institutional participation at reversal points.
# Designed for both bull and bear markets by trading reversals within the trend.
# Target: 80-150 total trades over 4 years (20-38/year) to stay within profitable range.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate RSI on 4h data
    def rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(len(close), np.nan)
        avg_loss = np.full(len(close), np.nan)
        
        # First average using simple mean
        if len(gain) >= period:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
            
            # Wilder smoothing
            for i in range(period+1, len(close)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_values = 100 - (100 / (1 + rs))
        return rsi_values
    
    rsi_values = rsi(close, 14)
    
    # Calculate EMA on daily close for trend filter
    def ema(close, period):
        ema_values = np.full(len(close), np.nan)
        if len(close) >= period:
            multiplier = 2 / (period + 1)
            ema_values[period-1] = np.mean(close[:period])
            for i in range(period, len(close)):
                ema_values[i] = (close[i] * multiplier) + (ema_values[i-1] * (1 - multiplier))
        return ema_values
    
    ema_50_1d = ema(df_1d['close'].values, 50)
    
    # Calculate volume moving average for confirmation
    def sma(arr, period):
        sma_values = np.full(len(arr), np.nan)
        for i in range(period-1, len(arr)):
            sma_values[i] = np.mean(arr[i-period+1:i+1])
        return sma_values
    
    volume_ma = sma(volume, 20)
    
    # Align all data to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), rsi_values)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, pd.DataFrame({'volume': volume}), volume_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(rsi_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi_aligned[i]
        ema_trend = ema_50_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol_current = volume[i]
        
        # Volume confirmation: current volume above average
        volume_confirmed = vol_current > vol_ma
        
        if position == 0:
            # Long setup: RSI oversold (<30) AND price above daily EMA (uptrend) AND volume confirmation
            if (rsi_val < 30 and 
                price > ema_trend and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short setup: RSI overbought (>70) AND price below daily EMA (downtrend) AND volume confirmation
            elif (rsi_val > 70 and 
                  price < ema_trend and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral (>50) OR price breaks below daily EMA
            if (rsi_val > 50 or 
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI returns to neutral (<50) OR price breaks above daily EMA
            if (rsi_val < 50 or 
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_RSI_MeanReversion_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0