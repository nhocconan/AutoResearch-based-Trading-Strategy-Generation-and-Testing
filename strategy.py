#!/usr/bin/env python3
"""
1d_1w_vwap_breakout_volume_v1
Hypothesis: VWAP breakout on daily timeframe with weekly trend filter and volume confirmation.
- Entry: Price breaks above/below VWAP (1d) with weekly trend alignment and volume > 1.5x 20-day average
- Trend filter: Weekly close > weekly EMA50 for longs, < for shorts
- Volume filter: Daily volume > 1.5x 20-period average to confirm institutional interest
- Exit: Price re-crosses VWAP or weekly trend reverses
- Position sizing: 0.25 long, -0.25 short
- Designed to capture institutional moves in both trending and ranging markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_vwap_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_up = close_1w > ema_50_1w
    trend_1w_down = close_1w < ema_50_1w
    
    # Forward fill trend
    trend_1w_up_series = pd.Series(trend_1w_up)
    trend_1w_down_series = pd.Series(trend_1w_down)
    trend_1w_up_ffilled = trend_1w_up_series.ffill().values
    trend_1w_down_ffilled = trend_1w_down_series.ffill().values
    
    # Align weekly trend to daily
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up_ffilled)
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down_ffilled)
    
    # Calculate daily VWAP
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.nancumsum(pv)
    cum_vol = np.nancumsum(volume)
    vwap = np.divide(cum_pv, cum_vol, out=np.zeros_like(cum_pv), where=cum_vol!=0)
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(vwap[i]) or np.isnan(volume_filter[i]) or
            np.isnan(close[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price crosses below VWAP OR weekly trend turns down
            if (close[i] < vwap[i]) or trend_1w_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: Price crosses above VWAP OR weekly trend turns up
            if (close[i] > vwap[i]) or trend_1w_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: Price crosses above VWAP + weekly uptrend + volume
            if (close[i] > vwap[i]) and trend_1w_up_aligned[i] and volume_filter[i]:
                # Confirm crossover from below
                if i > start_idx and close[i-1] <= vwap[i-1]:
                    position = 1
                    signals[i] = 0.25
            # Short entry: Price crosses below VWAP + weekly downtrend + volume
            elif (close[i] < vwap[i]) and trend_1w_down_aligned[i] and volume_filter[i]:
                # Confirm crossover from above
                if i > start_idx and close[i-1] >= vwap[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals