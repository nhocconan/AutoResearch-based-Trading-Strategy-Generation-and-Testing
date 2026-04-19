#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal with weekly trend filter and volume confirmation.
# Long when price breaks below S3 and closes back above S3 (mean reversion) AND weekly close > weekly open (bullish trend) AND volume > 1.3x 6h average volume.
# Short when price breaks above R3 and closes back below R3 (mean reversion) AND weekly close < weekly open (bearish trend) AND volume > 1.3x 6h average volume.
# Exit when price crosses the 6h VWAP (volume-weighted average price).
# Uses Camarilla for mean reversion levels, weekly trend for direction filter, volume for confirmation.
# Target: 15-30 trades/year per symbol.

name = "6h_Camarilla_WeeklyTrend_Volume_Reversion"
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
    
    # Get weekly data for trend filter (bullish/bearish based on open-close)
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_open = df_1w['open'].values
    weekly_bullish = weekly_close > weekly_open  # True if bullish week
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    # Get daily data for Camarilla levels (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    # Calculate Camarilla levels for each day using previous day's OHLC
    # H, L, C from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    # Camarilla formulas
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    R4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    S4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    # Align to 6s timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Get 6s average volume for confirmation (20-period)
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 6s VWAP for exit (typical price * volume cumulative)
    typical_price = (high + low + close) / 3
    vwap_num = pd.Series(typical_price * volume).rolling(window=50, min_periods=1).sum().values
    vwap_den = pd.Series(volume).rolling(window=50, min_periods=1).sum().values
    vwap = np.where(vwap_den != 0, vwap_num / vwap_den, typical_price)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_bullish_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(vol_ma_6h[i]) or np.isnan(vwap[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_6h[i]
        r3 = R3_aligned[i]
        s3 = S3_aligned[i]
        r4 = R4_aligned[i]
        s4 = S4_aligned[i]
        weekly_bull = weekly_bullish_aligned[i] > 0.5  # Convert back to boolean
        vwap_val = vwap[i]
        
        # Volume confirmation: at least 1.3x average volume
        vol_confirmed = vol > 1.3 * vol_ma
        
        if position == 0:
            # Long entry: price breaks below S3 but closes back above S3 (rejection of lower level)
            # Only in weekly bullish trend
            if price < s3 and close[i] > s3 and vol_confirmed and weekly_bull:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks above R3 but closes back below R3 (rejection of higher level)
            # Only in weekly bearish trend
            elif price > r3 and close[i] < r3 and vol_confirmed and not weekly_bull:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 6s VWAP
            if price < vwap_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 6s VWAP
            if price > vwap_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals