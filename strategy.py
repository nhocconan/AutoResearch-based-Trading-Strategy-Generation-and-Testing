#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1-day trend filter and volume confirmation
# The Alligator (Jaw/Teeth/Lips) identifies trend direction via SMAs
# Price above Lips = uptrend, Price below Lips = downtrend
# 1-day trend filter ensures alignment with higher timeframe trend
# Volume confirms participation in breakouts
# Target: 20-50 total trades over 4 years (5-12/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1-day EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Williams Alligator on 4h data
    # Jaw (blue): 13-period SMMA, shifted 8 bars
    # Teeth (red): 8-period SMMA, shifted 5 bars
    # Lips (green): 5-period SMMA, shifted 3 bars
    close = prices['close'].values
    
    # SMMA (Smoothed Moving Average) calculation
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CLOSE) / PERIOD
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift the lines as per Alligator definition
    jaw = np.roll(jaw, 8)  # shift 8 bars forward
    teeth = np.roll(teeth, 5)  # shift 5 bars forward
    lips = np.roll(lips, 3)  # shift 3 bars forward
    
    # Calculate 4h ATR for volatility filter and stop sizing
    high = prices['high'].values
    low = prices['low'].values
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Precompute hour of day for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Volume filter: 4h volume > 20-period average
    volume = prices['volume'].values
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(lips[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(atr_4h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter
        vol_filter = volume[i] > volume_ma_20[i]
        
        # Price levels
        price = close[i]
        ema50 = ema50_1d_aligned[i]
        lips_val = lips[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        
        if position == 0:
            # Long: price above Lips (bullish alignment) AND above daily EMA50 (uptrend filter) AND lips > teeth > jaw (perfect alignment) AND volume
            if price > lips_val and price > ema50 and lips_val > teeth_val and teeth_val > jaw_val and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price below Lips (bearish alignment) AND below daily EMA50 (downtrend filter) AND lips < teeth < jaw (perfect alignment) AND volume
            elif price < lips_val and price < ema50 and lips_val < teeth_val and teeth_val < jaw_val and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: stop loss (2x ATR below entry) or price crosses below Lips
            if price <= entry_price - 2.0 * atr_4h[i] or price < lips_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss (2x ATR above entry) or price crosses above Lips
            if price >= entry_price + 2.0 * atr_4h[i] or price > lips_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dTrendFilter_Volume"
timeframe = "4h"
leverage = 1.0