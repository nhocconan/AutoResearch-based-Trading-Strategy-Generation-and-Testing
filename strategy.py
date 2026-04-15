# 1d_1W_RSI_Turtle_Smooth
# Hypothesis: Use weekly RSI as a trend filter (RSI > 50 = bullish bias, < 50 = bearish bias) combined with daily price action.
# Enter long when price breaks above 20-day high with bullish weekly RSI and volume confirmation.
# Enter short when price breaks below 20-day low with bearish weekly RSI and volume confirmation.
# Exit when price crosses the 10-day EMA in the opposite direction.
# This strategy aims to capture medium-term trends while avoiding choppy markets via weekly RSI filter.
# Weekly RSI provides a smoother, more reliable trend signal than daily indicators, reducing whipsaws.
# Target: 20-40 trades/year per symbol with disciplined entries.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 10-day EMA for exit signal
    close_s = pd.Series(close)
    ema_10 = close_s.ewm(span=10, adjust=False, min_periods=10).values
    
    # Get weekly data for RSI filter
    weekly = get_htf_data(prices, '1w')
    if len(weekly) < 14:
        return np.zeros(n)
    
    # Calculate weekly RSI(14)
    weekly_close = weekly['close'].values
    delta = np.diff(weekly_close, prepend=weekly_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_weekly = 100 - (100 / (1 + rs))
    rsi_weekly_aligned = align_htf_to_ltf(prices, weekly, rsi_weekly)
    
    # Daily Donchian channels (20-period) for entry
    daily = get_htf_data(prices, '1d')
    if len(daily) < 20:
        return np.zeros(n)
    
    daily_high = daily['high'].values
    daily_low = daily['low'].values
    daily_close = daily['close'].values
    
    # Calculate Donchian channels
    donch_high = np.full(len(daily_high), np.nan)
    donch_low = np.full(len(daily_low), np.nan)
    for i in range(20, len(daily_high)):
        donch_high[i] = np.max(daily_high[i-20:i])
        donch_low[i] = np.min(daily_low[i-20:i])
    
    donch_high_aligned = align_htf_to_ltf(prices, daily, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, daily, donch_low)
    
    # Volume filter: 1.5x 20-day average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_threshold = 1.5 * vol_ma
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(rsi_weekly_aligned[i]) or np.isnan(vol_threshold[i])):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # Long conditions: price above Donchian high, bullish weekly RSI (>50), volume spike
        long_condition = (close[i] > donch_high_aligned[i] and 
                         rsi_weekly_aligned[i] > 50 and 
                         volume[i] > vol_threshold[i])
        
        # Short conditions: price below Donchian low, bearish weekly RSI (<50), volume spike
        short_condition = (close[i] < donch_low_aligned[i] and 
                          rsi_weekly_aligned[i] < 50 and 
                          volume[i] > vol_threshold[i])
        
        # Exit conditions: price crosses 10-day EMA in opposite direction
        exit_long = (i > 0 and signals[i-1] > 0 and close[i] < ema_10[i])
        exit_short = (i > 0 and signals[i-1] < 0 and close[i] > ema_10[i])
        
        if long_condition:
            signals[i] = 0.25
        elif short_condition:
            signals[i] = -0.25
        elif exit_long or exit_short:
            signals[i] = 0.0
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_1W_RSI_Turtle_Smooth"
timeframe = "1d"
leverage = 1.0