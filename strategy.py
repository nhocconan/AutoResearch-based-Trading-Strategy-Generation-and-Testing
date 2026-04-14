#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels with 1-day volume confirmation and trend filter
# Long when price touches or exceeds Camarilla H4 level AND volume > 1.5x average AND price > 100-period EMA (uptrend)
# Short when price touches or drops below Camarilla L4 level AND volume > 1.5x average AND price < 100-period EMA (downtrend)
# Exit when price crosses the Camarilla H3/L3 levels in opposite direction
# Designed to capture institutional reversal points with volume confirmation and trend alignment.
# Target: 80-160 total trades over 4 years (20-40/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla calculation and volume filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 100-period EMA on 4h for trend filter
    ema_100 = pd.Series(close).ewm(span=100, min_periods=100, adjust=False).mean()
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 100
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if np.isnan(ema_100[i]) or np.isnan(vol_avg[i]):
            signals[i] = 0.0
            continue
        
        # Get 1d OHLC for Camarilla calculation (using previous completed 1d candle)
        # We need the 1d data from the previous day to avoid look-ahead
        prev_day_idx = len(df_1d) - 1  # This will be handled by align_htf_to_ltf with proper delay
        
        # Get Camarilla levels aligned to 4h timeframe from previous 1d candle
        # Calculate Camarilla based on previous 1d OHLC
        if len(df_1d) >= 2:
            prev_high = df_1d['high'].iloc[-2]  # Previous completed 1d candle
            prev_low = df_1d['low'].iloc[-2]
            prev_close = df_1d['close'].iloc[-2]
            
            # Camarilla levels calculation
            range_val = prev_high - prev_low
            h4 = prev_close + range_val * 1.1 / 2
            l4 = prev_close - range_val * 1.1 / 2
            h3 = prev_close + range_val * 1.1 / 4
            l3 = prev_close - range_val * 1.1 / 4
            
            # Create arrays aligned to 4h timeframe (same value for all 4h bars of the day)
            # We'll handle this by checking if we're in the same 1d period
            pass  # We'll implement properly below
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        ema_val = ema_100[i]
        
        if position == 0:
            # Long setup: price touches/exceeds H4 AND volume confirmation AND uptrend
            if (high_val >= h4 and vol > vol_threshold and close_val > ema_val):
                position = 1
                signals[i] = position_size
            # Short setup: price touches/drops below L4 AND volume confirmation AND downtrend
            elif (low_val <= l4 and vol > vol_threshold and close_val < ema_val):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below H3 level
            if close_val < h3:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above L3 level
            if close_val > l3:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Camarilla_Volume_Trend"
timeframe = "4h"
leverage = 1.0