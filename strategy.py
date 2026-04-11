#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla breakout with 1w trend filter and volume confirmation
# Long when price breaks above Camarilla H4 level + volume > 1.5x 10-day avg + 1w trend up
# Short when price breaks below Camarilla L4 level + volume > 1.5x 10-day avg + 1w trend down
# Exit when price returns to Camarilla P (pivot) level or 1w trend reverses
# Designed for 8-15 trades/year on 1d timeframe with strong trend capture and low turnover

name = "1d_1w_camarilla_breakout_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1w EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 10-day average volume for volume filter
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    # Calculate Camarilla levels from previous day
    # Camarilla formulas:
    # H4 = C + 1.1/2 * (H - L)
    # L4 = C - 1.1/2 * (H - L)
    # P = (H + L + C) / 3
    # We use previous day's H, L, C to calculate today's levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # Set first value to avoid roll issues
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_H4 = prev_close + 1.1/2 * (prev_high - prev_low)
    camarilla_L4 = prev_close - 1.1/2 * (prev_high - prev_low)
    camarilla_P = (prev_high + prev_low + prev_close) / 3
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(10, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_H4[i]) or np.isnan(camarilla_L4[i]) or 
            np.isnan(camarilla_P[i]) or np.isnan(vol_ma_10[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 10-day average
        volume_filter = volume[i] > 1.5 * vol_ma_10[i]
        
        # Trend filter: price relative to 1w EMA20
        is_uptrend = close[i] > ema_20_1w_aligned[i]
        is_downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Entry conditions
        camarilla_breakout_up = close[i] > camarilla_H4[i]  # Break above H4
        camarilla_breakdown_down = close[i] < camarilla_L4[i]  # Break below L4
        
        long_entry = camarilla_breakout_up and volume_filter and is_uptrend
        short_entry = camarilla_breakdown_down and volume_filter and is_downtrend
        
        # Exit conditions
        long_exit = close[i] < camarilla_P[i]  # Return to pivot level
        short_exit = close[i] > camarilla_P[i]  # Return to pivot level
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals