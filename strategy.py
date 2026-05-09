#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_RSI4_Divergence_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and RSI
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 14:
        return np.zeros(n)
    
    # Daily close for RSI calculation
    daily_close = df_d['close'].values
    
    # Calculate RSI(4) on daily timeframe
    close_series = pd.Series(daily_close)
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/4, min_periods=4, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/4, min_periods=4, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align RSI to 6h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_d, rsi_values)
    
    # Daily EMA(50) for trend filter
    ema50_d = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_d_aligned = align_htf_to_ltf(prices, df_d, ema50_d)
    
    # Volume confirmation: current volume > 2.0x 24-period average (4 days)
    vol_series = pd.Series(volume)
    vol_ma24 = vol_series.rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi_aligned[i]) or np.isnan(ema50_d_aligned[i]) or 
            np.isnan(vol_ma24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma24[i]
        
        if position == 0:
            # Bullish divergence: price makes lower low, RSI makes higher low
            # Look back 3 periods for simplicity
            if i >= 3:
                price_lower_low = low[i] < low[i-1] and low[i-1] < low[i-2]
                rsi_higher_low = rsi_aligned[i] > rsi_aligned[i-1] and rsi_aligned[i-1] > rsi_aligned[i-2]
                
                # Bearish divergence: price makes higher high, RSI makes lower high
                price_higher_high = high[i] > high[i-1] and high[i-1] > high[i-2]
                rsi_lower_high = rsi_aligned[i] < rsi_aligned[i-1] and rsi_aligned[i-1] < rsi_aligned[i-2]
                
                # Long: bullish divergence + above daily EMA trend + volume
                if price_lower_low and rsi_higher_low and close[i] > ema50_d_aligned[i] and vol_ok:
                    signals[i] = 0.25
                    position = 1
                # Short: bearish divergence + below daily EMA trend + volume
                elif price_higher_high and rsi_lower_high and close[i] < ema50_d_aligned[i] and vol_ok:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: RSI crosses below 50 (momentum loss) or price closes below EMA
            if rsi_aligned[i] < 50 or close[i] < ema50_d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI crosses above 50 or price closes above EMA
            if rsi_aligned[i] > 50 or close[i] > ema50_d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals