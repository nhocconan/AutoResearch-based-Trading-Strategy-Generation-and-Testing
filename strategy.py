# This strategy implements a 6H Fibonacci extension bounce system with weekly trend filter
# Long entries occur when price retraces to Fibonacci levels (38.2%, 50%, 61.8%) during a weekly uptrend
# Short entries occur at the same levels during a weekly downtrend
# Uses volume confirmation to filter out low-probability retracements
# Target: 50-150 trades over 4 years on 6H timeframe
# Designed to work in both bull and bear markets by following weekly trend direction

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === Weekly EMA50 for trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 6H price data for swing detection ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === Swing high/low detection (20-period lookback) ===
    # Swing high: highest high in 20 bars
    rolling_max_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Swing low: lowest low in 20 bars
    rolling_min_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume confirmation ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(rolling_max_high[i]) or 
            np.isnan(rolling_min_low[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        weekly_trend = ema_50_1w_aligned[i]
        swing_high = rolling_max_high[i]
        swing_low = rolling_min_low[i]
        vol_ratio_val = vol_ratio[i]
        
        # Calculate Fibonacci retracement levels from recent swing
        price_range = swing_high - swing_low
        if price_range <= 0:  # Avoid division by zero
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Fibonacci levels: 38.2%, 50%, 61.8% retracement
        fib_382 = swing_high - (price_range * 0.382)
        fib_500 = swing_high - (price_range * 0.500)
        fib_618 = swing_high - (price_range * 0.618)
        
        # Tolerance for level touches (0.5% of price)
        tolerance = price_close * 0.005
        
        if position == 0:
            # Long setup: weekly uptrend + price at Fibonacci support + volume
            if (price_close > weekly_trend and  # Weekly uptrend
                vol_ratio_val > 1.3):           # Volume confirmation
                
                # Check if price is near any Fibonacci support level
                near_fib = (abs(price_close - fib_382) < tolerance or
                           abs(price_close - fib_500) < tolerance or
                           abs(price_close - fib_618) < tolerance)
                
                if near_fib:
                    signals[i] = 0.25
                    position = 1
            
            # Short setup: weekly downtrend + price at Fibonacci resistance + volume
            elif (price_close < weekly_trend and  # Weekly downtrend
                  vol_ratio_val > 1.3):           # Volume confirmation
                
                # Check if price is near any Fibonacci resistance level
                near_fib = (abs(price_close - fib_382) < tolerance or
                           abs(price_close - fib_500) < tolerance or
                           abs(price_close - fib_618) < tolerance)
                
                if near_fib:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            if position == 1:
                # Exit long: weekly trend turns down OR price breaks above swing high (failed bounce)
                if (price_close < weekly_trend or price_close > swing_high):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25  # Hold long
            
            elif position == -1:
                # Exit short: weekly trend turns up OR price breaks below swing low (failed bounce)
                if (price_close > weekly_trend or price_close < swing_low):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals

name = "6H_Fibonacci_Bounce_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0