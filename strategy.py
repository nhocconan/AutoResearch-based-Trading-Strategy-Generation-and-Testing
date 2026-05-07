#!/usr/bin/env python3
# 4h_RSI_Div_Bull_Bear_30mVolume
# Hypothesis: RSI divergence on 30m with 4h trend filter and volume spike. 
# Bullish: price makes lower low but RSI makes higher low on 30m, 4h EMA50 up, volume spike.
# Bearish: price makes higher high but RSI makes lower high on 30m, 4h EMA50 down, volume spike.
# Exit: RSI crosses 50 in opposite direction. Designed for low frequency and high win rate.

name = "4h_RSI_Div_Bull_Bear_30mVolume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def rsi(series, period=14):
    delta = np.diff(series, prepend=series[0])
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(up).ewm(alpha=1/period, adjust=False).mean()
    roll_down = pd.Series(down).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-10)
    return (100 - (100 / (1 + rs))).values

def find_divergence(price, rsi_vals, lookback=14):
    # Bullish divergence: price lower low, RSI higher low
    # Bearish divergence: price higher high, RSI lower high
    min_price_idx = np.argmin(price[-lookback:])
    max_price_idx = np.argmax(price[-lookback:])
    min_rsi_idx = np.argmin(rsi_vals[-lookback:])
    max_rsi_idx = np.argmax(rsi_vals[-lookback:])
    
    bull_div = (price[-lookback:][min_price_idx] < price[-lookback:][0] and 
                rsi_vals[-lookback:][min_rsi_idx] > rsi_vals[-lookback:][0])
    bear_div = (price[-lookback:][max_price_idx] > price[-lookback:][0] and 
                rsi_vals[-lookback:][max_rsi_idx] < rsi_vals[-lookback:][0])
    return bull_div, bear_div

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 30m data for RSI and divergence
    df_30m = get_htf_data(prices, '30m')
    if len(df_30m) == 0:
        return np.zeros(n)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) == 0:
        return np.zeros(n)
    
    # Calculate RSI on 30m close
    rsi_30m = rsi(df_30m['close'], 14)
    rsi_30m_aligned = align_ltf_to_htf(prices, df_30m, rsi_30m)
    
    # Calculate EMA50 on 4h close for trend
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_ltf_to_htf(prices, df_4h, ema50_4h)
    
    # Volume spike: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 50, 20)  # 30m RSI, 4h EMA50, vol MA
    
    for i in range(start_idx, n):
        # Align indices for 30m and 4h data
        idx_30m = i // 2  # 2 x 30m = 1h, 4 x 30m = 2h, 8 x 30m = 4h
        idx_4h = i // 16  # 16 x 15m = 4h
        
        if idx_30m < 14 or idx_4h < 50:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get window for divergence check
        price_window = df_30m['close'].values[max(0, idx_30m-14):idx_30m+1]
        rsi_window = rsi_30m_aligned[max(0, i-28):i+1:2]  # Every other 15m bar = 30m
        
        if len(price_window) < 15 or len(rsi_window) < 15:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bull_div, bear_div = find_divergence(price_window, rsi_window, 14)
        
        if position == 0:
            # Long: bullish divergence, price above EMA50 (uptrend), volume spike
            if (bull_div and 
                close[i] > ema50_4h_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish divergence, price below EMA50 (downtrend), volume spike
            elif (bear_div and 
                  close[i] < ema50_4h_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI crosses below 50
            if rsi_30m_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI crosses above 50
            if rsi_30m_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Note: align_ltf_to_htf function is used as inverse of align_htf_to_ltf
# If not available, we'll implement it or use align_htf_to_ltf with inverse logic
# For simplicity in this context, assuming it exists or we adjust indices accordingly
# In practice, we would use align_htf_to_ltf and access by index directly
# Let's correct the approach to use align_htf_to_ltf properly

#!/usr/bin/env python3
# 4h_RSI_Div_Bull_Bear_30mVolume
# Hypothesis: RSI divergence on 30m with 4h trend filter and volume spike. 
# Bullish: price makes lower low but RSI makes higher low on 30m, 4h EMA50 up, volume spike.
# Bearish: price makes higher high but RSI makes lower high on 30m, 4h EMA50 down, volume spike.
# Exit: RSI crosses 50 in opposite direction. Designed for low frequency and high win rate.

name = "4h_RSI_Div_Bull_Bear_30mVolume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def rsi(series, period=14):
    delta = np.diff(series, prepend=series[0])
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(up).ewm(alpha=1/period, adjust=False).mean()
    roll_down = pd.Series(down).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-10)
    return (100 - (100 / (1 + rs))).values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 30m data for RSI and price
    df_30m = get_htf_data(prices, '30m')
    if len(df_30m) == 0:
        return np.zeros(n)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) == 0:
        return np.zeros(n)
    
    # Calculate RSI on 30m close
    rsi_30m = rsi(df_30m['close'], 14)
    rsi_30m_aligned = align_htf_to_ltf(prices, df_30m, rsi_30m)
    
    # Get 30m close for price action
    close_30m = df_30m['close'].values
    close_30m_aligned = align_htf_to_ltf(prices, df_30m, close_30m)
    
    # Calculate EMA50 on 4h close for trend
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume spike: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 50, 20)  # Need data for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(rsi_30m_aligned[i]) or np.isnan(close_30m_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for RSI divergence on 30m chart (need to look back)
            # We'll check last few 30m bars for divergence
            lookback = 6  # Last few 30m periods
            start_look = max(0, i - lookback*2)  # Each 30m bar = 2x 15m bars
            
            # Extract windows for divergence check
            price_window = close_30m_aligned[start_look:i+1:2]  # Every other = 30m
            rsi_window = rsi_30m_aligned[start_look:i+1:2]
            
            if len(price_window) >= 5 and len(rsi_window) >= 5:
                # Find recent lows and highs
                price_min_idx = np.argmin(price_window[-5:])
                price_max_idx = np.argmax(price_window[-5:])
                rsi_min_idx = np.argmin(rsi_window[-5:])
                rsi_max_idx = np.argmax(rsi_window[-5:])
                
                # Current values (most recent in window)
                curr_price = price_window[-1]
                curr_rsi = rsi_window[-1]
                prev_min_price = price_window[-5:][price_min_idx] if len(price_window) >=5 else price_window[0]
                prev_max_price = price_window[-5:][price_max_idx] if len(price_window) >=5 else price_window[0]
                prev_min_rsi = rsi_window[-5:][rsi_min_idx] if len(rsi_window) >=5 else rsi_window[0]
                prev_max_rsi = rsi_window[-5:][rsi_max_idx] if len(rsi_window) >=5 else rsi_window[0]
                
                bull_div = (curr_price < prev_min_price and curr_rsi > prev_min_rsi)
                bear_div = (curr_price > prev_max_price and curr_rsi < prev_max_rsi)
            else:
                bull_div = bear_div = False
            
            # Long: bullish divergence, price above EMA50 (uptrend), volume spike
            if (bull_div and 
                close[i] > ema50_4h_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish divergence, price below EMA50 (downtrend), volume spike
            elif (bear_div and 
                  close[i] < ema50_4h_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI crosses below 50
            if rsi_30m_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI crosses above 50
            if rsi_30m_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals