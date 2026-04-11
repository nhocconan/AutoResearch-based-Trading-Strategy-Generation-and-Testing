#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and session timing
# - Uses 4h EMA(21) for trend direction (long when price > EMA, short when price < EMA)
# - Enters on 1h Camarilla H3/L3 breakout in direction of 4h trend
# - Only trades during 08-20 UTC session to avoid low-volume periods
# - Camarilla levels calculated from previous 1h bar's range
# - Discrete position sizing: ±0.20 to limit drawdown and reduce fee churn
# - Target: 15-37 trades/year (60-150 total over 4 years) to stay within fee drag limits
# - Works in bull markets (trend-following breakouts) and bear markets (trend-following breakdowns)

name = "1h_4h_camarilla_trend_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_time = prices['open_time']
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute session hours ONCE before loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return signals
    
    # Pre-compute 4h EMA(21) for trend
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if EMA data is invalid
        if np.isnan(ema_21_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine 4h trend direction
        trend_long = close[i] > ema_21_4h_aligned[i]   # Price above 4h EMA = uptrend
        trend_short = close[i] < ema_21_4h_aligned[i]  # Price below 4h EMA = downtrend
        
        # Calculate Camarilla levels from previous 1h bar
        # Camarilla: based on previous bar's range
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        range_val = prev_high - prev_low
        
        if range_val <= 0:
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
            continue
        
        # Camarilla levels (H3/L3 are key breakout levels)
        camarilla_h3 = prev_close + range_val * 1.1 / 4
        camarilla_l3 = prev_close - range_val * 1.1 / 4
        
        # Entry conditions: breakout in direction of 4h trend
        enter_long = trend_long and close[i] > camarilla_h3
        enter_short = trend_short and close[i] < camarilla_l3
        
        # Exit conditions: reversal of 4h trend
        exit_long = not trend_long
        exit_short = not trend_short
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.20
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals