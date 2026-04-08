#!/usr/bin/env python3
# 1h_price_action_volume_momentum_v1
# Hypothesis: 1h strategy using price action (close > open) and volume surge for momentum entries.
# Filtered by 4h trend (EMA50) and 1h RSI to avoid reversals. Long when bullish candle with volume > 1.5x average and price above 4h EMA50.
# Short when bearish candle with volume surge and price below 4h EMA50. Uses 1h RSI (50) to avoid overextended moves.
# Designed for 15-35 trades/year on 1h to avoid fee drag. Works in bull/bear via 4h trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_price_action_volume_momentum_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1h RSI(14) for momentum filter
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(prices, np.nan)
        avg_loss = np.full_like(prices, np.nan)
        
        # First average
        if len(gain) >= period:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
        
        # Wilder's smoothing
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Volume average (20-period)
    vol_avg = np.full_like(volume, np.nan)
    for i in range(20, n):
        vol_avg[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(50, 20, 14)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema50_4h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_avg[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Bullish/bearish candle
        bullish_candle = close[i] > open_price[i]
        bearish_candle = close[i] < open_price[i]
        
        # Volume surge condition
        volume_surge = volume[i] > 1.5 * vol_avg[i]
        
        if position == 1:  # Long position
            # Exit: bearish candle with volume surge or RSI < 40 (momentum loss)
            if bearish_candle and volume_surge:
                position = 0
                signals[i] = 0.0
            elif rsi[i] < 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: bullish candle with volume surge or RSI > 60 (momentum loss)
            if bullish_candle and volume_surge:
                position = 0
                signals[i] = 0.0
            elif rsi[i] > 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: bullish candle with volume surge, price above 4h EMA50, RSI > 50
            if (bullish_candle and 
                volume_surge and 
                close[i] > ema50_4h_aligned[i] and 
                rsi[i] > 50):
                position = 1
                signals[i] = 0.20
            # Short entry: bearish candle with volume surge, price below 4h EMA50, RSI < 50
            elif (bearish_candle and 
                  volume_surge and 
                  close[i] < ema50_4h_aligned[i] and 
                  rsi[i] < 50):
                position = -1
                signals[i] = -0.20
    
    return signals