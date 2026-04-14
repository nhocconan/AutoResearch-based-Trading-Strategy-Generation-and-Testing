#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band breakout with daily RSI filter and volume confirmation
# Bollinger Bands provide dynamic volatility-based support/resistance. Breakouts with volume
# and daily RSI filter (RSI > 60 for longs, RSI < 40 for shorts) capture strong momentum
# while avoiding false breakouts in low volatility. Works in bull/bear by using RSI to
# filter for momentum direction. Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate daily RSI(14) for trend filter
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_14_1d = calculate_rsi(close_1d, 14)
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Calculate Bollinger Bands (20, 2) on 4h
    bb_period = 20
    bb_std = 2
    close_series = pd.Series(close)
    bb_mid = close_series.rolling(window=bb_period, min_periods=bb_period).mean()
    bb_std_dev = close_series.rolling(window=bb_period, min_periods=bb_period).std()
    bb_upper = bb_mid + (bb_std_dev * bb_std)
    bb_lower = bb_mid - (bb_std_dev * bb_std)
    bb_mid = bb_mid.values
    bb_upper = bb_upper.values
    bb_lower = bb_lower.values
    
    # Volume confirmation: volume > 1.5x average volume (24-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=24, min_periods=24).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(bb_period, 24) + 1
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(rsi_14_1d_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above upper BB with volume filter AND daily RSI > 60
            if (price > bb_upper[i] and vol > 1.5 * avg_vol[i] and 
                rsi_14_1d_aligned[i] > 60):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower BB with volume filter AND daily RSI < 40
            elif (price < bb_lower[i] and vol > 1.5 * avg_vol[i] and 
                  rsi_14_1d_aligned[i] < 40):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below middle BB
            if price < bb_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above middle BB
            if price > bb_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Bollinger_Breakout_Daily_RSI_Volume"
timeframe = "4h"
leverage = 1.0