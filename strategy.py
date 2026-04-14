#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Volume-weighted RSI with 1-day Bollinger Band squeeze filter
# Long when RSI < 30 AND price above 20-period SMA AND Bollinger Band width < 50th percentile (squeeze) AND volume > 1.5x average
# Short when RSI > 70 AND price below 20-period SMA AND Bollinger Band width < 50th percentile (squeeze) AND volume > 1.5x average
# Exit when RSI crosses 50 in opposite direction
# RSI identifies overbought/oversold conditions, Bollinger squeeze indicates low volatility primed for breakout,
# volume confirms institutional participation. Designed to work in both trending and ranging markets.
# Target: 80-180 total trades over 4 years (20-45/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Bollinger Band squeeze filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate RSI on 4h (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 20-period SMA on 4h
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean()
    
    # Calculate Bollinger Bands on 1d (20-period, 2 std)
    close_1d = df_1d['close'].values
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean()
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std()
    upper_bb = sma_20_1d + (2 * std_20_1d)
    lower_bb = sma_20_1d - (2 * std_20_1d)
    bb_width = upper_bb - lower_bb
    
    # Calculate Bollinger Band width percentile (50-period lookback for median)
    bb_width_median = pd.Series(bb_width).rolling(window=50, min_periods=50).median()
    bb_squeeze = bb_width < bb_width_median  # True when width below median (squeeze condition)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (max of 20 for RSI/SMA + buffer)
    start = 40
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi[i]) or np.isnan(sma_20[i]) or 
            np.isnan(bb_squeeze[i]) if i < len(bb_squeeze) else True or
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        # Get Bollinger squeeze value aligned to 4h timeframe
        bb_squeeze_aligned = align_htf_to_ltf(prices, df_1d, bb_squeeze.astype(float))
        squeeze_active = bb_squeeze_aligned[i] > 0.5  # True if squeeze condition
        
        rsi_val = rsi[i]
        sma_val = sma_20[i]
        close_val = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: RSI < 30 (oversold) AND price above SMA AND Bollinger squeeze AND volume confirmation
            if (rsi_val < 30 and close_val > sma_val and squeeze_active and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: RSI > 70 (overbought) AND price below SMA AND Bollinger squeeze AND volume confirmation
            elif (rsi_val > 70 and close_val < sma_val and squeeze_active and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI crosses above 50
            if rsi_val > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI crosses below 50
            if rsi_val < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_VolumeWeighted_RSI_BB_Squeeze"
timeframe = "4h"
leverage = 1.0