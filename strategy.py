#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and volume confirmation.
# Long when RSI(14) < 30 (oversold), price > 4h EMA50 (bullish trend), and volume > 1.5x average.
# Short when RSI(14) > 70 (overbought), price < 4h EMA50 (bearish trend), and volume > 1.5x average.
# Exit when RSI returns to neutral (40-60 range) or trend reverses.
# Uses 4h for trend direction, 1h for entry timing and mean reversion signals.
# Designed for ~20-30 trades/year with strict entry conditions to avoid overtrading.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate RSI(14) on 1h closes
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use exponential moving average for RSI calculation
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    rsi = np.zeros(n)
    
    # Initialize first values
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        
        # Calculate subsequent values using EMA formula
        for i in range(14, len(gain)):
            avg_gain[i] = (gain[i] * 1 + avg_gain[i-1] * 13) / 14
            avg_loss[i] = (loss[i] * 1 + avg_loss[i-1] * 13) / 14
            
            if avg_loss[i] != 0:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i+1] = 100 - (100 / (1 + rs))
            else:
                rsi[i+1] = 100
    
    # Calculate 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # 20% position size
    
    # Warmup: need RSI and volume MA
    start_idx = 34  # RSI needs 14+20 periods to stabilize
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_now = rsi[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filters from 4h EMA50
        bullish_trend = price > ema50_4h_aligned[i]
        bearish_trend = price < ema50_4h_aligned[i]
        
        if position == 0:
            # Long: RSI oversold (<30) with volume and bullish trend
            if rsi_now < 30 and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: RSI overbought (>70) with volume and bearish trend
            elif rsi_now > 70 and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral (>40) or trend turns bearish
            if rsi_now > 40 or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI returns to neutral (<60) or trend turns bullish
            if rsi_now < 60 or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_RSI_MeanReversion_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0