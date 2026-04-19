# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d RSI trend filter and volume confirmation.
# Long when price breaks above R3 and 1d RSI > 55 and volume > 1.5x 6h average volume.
# Short when price breaks below S3 and 1d RSI < 45 and volume > 1.5x 6h average volume.
# Exit when price crosses back below/above H4/L4.
# Uses Camarilla pivots for intraday support/resistance, RSI for trend filter, volume for confirmation.
# Target: 15-35 trades/year per symbol (60-140 total over 4 years).
name = "6h_Camarilla_RSI_Volume"
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
    
    # Calculate Camarilla pivot levels from previous day
    # Using previous day's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # First value: use current day's values (no previous day available)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    R4 = prev_close + range_hl * 1.500
    R3 = prev_close + range_hl * 1.250
    R2 = prev_close + range_hl * 1.166
    R1 = prev_close + range_hl * 1.083
    S1 = prev_close - range_hl * 1.083
    S2 = prev_close - range_hl * 1.166
    S3 = prev_close - range_hl * 1.250
    S4 = prev_close - range_hl * 1.500
    H4 = R3  # Commonly used exit level
    L4 = S3
    
    # Get daily data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate RSI (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Wilder's smoothing for RSI
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    avg_gain = wilder_smooth(gain, period)
    avg_loss = wilder_smooth(loss, period)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align RSI to 6h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Get 6h average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure volume MA and pivots are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(R3[i]) or np.isnan(S3[i]) or np.isnan(H4[i]) or np.isnan(L4[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi = rsi_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: price breaks above R3, RSI > 55, volume confirmation
            if price > R3[i] and rsi > 55 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3, RSI < 45, volume confirmation
            elif price < S3[i] and rsi < 45 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below H4 (R3)
            if price < H4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above L4 (S3)
            if price > L4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals