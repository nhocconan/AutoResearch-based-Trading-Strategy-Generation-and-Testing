#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull Power + Bear Power with 1d trend filter
# Bull Power = High - EMA13, Bear Power = EMA13 - Low
# Long when Bull Power > 0 and Bear Power < 0 (bullish momentum) + 1d EMA50 uptrend
# Short when Bear Power > 0 and Bull Power < 0 (bearish momentum) + 1d EMA50 downtrend
# Uses volume confirmation to avoid false signals. Works in bull (trend continuation) and bear (mean reversion via trend filter).
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
name = "6h_ElderRay_BullPower_1dTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA13 for Elder Ray (using 13-period EMA)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).values
    
    # Calculate Bull Power and Bear Power
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low
    
    # Calculate daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).values
    
    # Align daily EMA50 to 6h (wait for daily close)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.3 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        ema50_val = ema50_1d_aligned[i]
        vol_filter = volume_filter[i]
        close_val = close[i]
        
        if position == 0:
            # Long: Bull Power > 0 (bullish momentum) AND Bear Power < 0 (not bearish) + 1d uptrend + volume
            if bull_val > 0 and bear_val < 0 and close_val > ema50_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 (bearish momentum) AND Bull Power < 0 (not bullish) + 1d downtrend + volume
            elif bear_val > 0 and bull_val < 0 and close_val < ema50_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: momentum shifts bearish or trend breaks
            if bull_val <= 0 or bear_val >= 0 or close_val < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: momentum shifts bullish or trend breaks
            if bear_val <= 0 or bull_val >= 0 or close_val > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals