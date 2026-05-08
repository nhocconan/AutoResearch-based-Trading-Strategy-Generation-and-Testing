#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) momentum with 4h EMA(50) trend filter and volume confirmation
# Long when RSI > 50 crossing up + price > 4h EMA50 + volume > 1.5x 20-period EMA of volume
# Short when RSI < 50 crossing down + price < 4h EMA50 + volume > 1.5x 20-period EMA of volume
# Uses 4h for trend direction and volume confirmation, 1h only for entry timing via RSI momentum
# Designed for 1h timeframe to target 20-40 trades/year (80-160 total over 4 years)
# RSI momentum captures short-term swings while higher timeframe filters prevent counter-trend trades

name = "1h_RSI_Momentum_4hEMA50_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend and volume filters
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50 = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h volume EMA (20-period) for volume filter
    vol_ema_20 = pd.Series(df_4h['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 4h indicators to 1h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ema_20)
    
    # Calculate 1h RSI(14) for momentum signals
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(ema_50_aligned[i]) or np.isnan(vol_ema_20_aligned[i]) or \
           np.isnan(rsi_values[i]) or np.isnan(rsi_values[i-1]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 4h volume > 1.5x 20-period EMA
        # Find the most recent completed 4h bar
        idx_4h = 0
        while idx_4h < len(df_4h) and df_4h.iloc[idx_4h]['open_time'] <= prices.iloc[i]['open_time']:
            idx_4h += 1
        idx_4h -= 1  # last completed 4h bar
        
        if idx_4h < 0:
            vol_filter = False
        else:
            vol_4h_current = df_4h.iloc[idx_4h]['volume']
            vol_filter = vol_4h_current > 1.5 * vol_ema_20_aligned[i]
        
        if position == 0:
            # Look for entry: RSI crossing 50 + trend + volume
            rsi_cross_up = rsi_values[i-1] < 50 and rsi_values[i] >= 50
            rsi_cross_down = rsi_values[i-1] > 50 and rsi_values[i] <= 50
            
            long_condition = rsi_cross_up and close[i] > ema_50_aligned[i] and vol_filter
            short_condition = rsi_cross_down and close[i] < ema_50_aligned[i] and vol_filter
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: RSI crosses below 50 or price crosses below EMA50
            rsi_cross_down = rsi_values[i-1] >= 50 and rsi_values[i] < 50
            if rsi_cross_down or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: RSI crosses above 50 or price crosses above EMA50
            rsi_cross_up = rsi_values[i-1] <= 50 and rsi_values[i] > 50
            if rsi_cross_up or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals