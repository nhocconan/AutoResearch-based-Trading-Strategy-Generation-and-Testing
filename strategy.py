#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Daily ATR Breakout with Volume and Trend Filter
# Hypothesis: Price breaking out of daily ATR-based channels with volume confirmation
# and trend filter (price vs 200 EMA) works in both bull and bear markets.
# In bull markets: buy on upward breakouts, sell on downward breakdowns.
# In bear markets: sell on downward breakdowns, buy on upward breakouts.
# Target: 12-37 trades/year (50-150 over 4 years).

name = "12h_daily_atr_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR and ATR-based channels
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 1:
        return np.zeros(n)
    
    # Calculate daily ATR (14-period)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # True Range calculation
    tr1 = high_daily - low_daily
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR calculation
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14  # Wilder's smoothing
    
    # ATR-based channels (using previous day's data)
    upper_channel = close_daily + (atr * 2.0)
    lower_channel = close_daily - (atr * 2.0)
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    upper_channel = np.roll(upper_channel, 1)
    lower_channel = np.roll(lower_channel, 1)
    atr = np.roll(atr, 1)
    
    # Handle first element
    if len(upper_channel) > 1:
        upper_channel[0] = upper_channel[1]
        lower_channel[0] = lower_channel[1]
        atr[0] = atr[1]
    else:
        upper_channel[0] = 0
        lower_channel[0] = 0
        atr[0] = 0
    
    # Align to 12h timeframe
    upper_channel_aligned = align_htf_to_ltf(prices, df_daily, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_daily, lower_channel)
    atr_aligned = align_htf_to_ltf(prices, df_daily, atr)
    
    # Trend filter: price vs 200 EMA
    close_series = pd.Series(close)
    ema_200 = close_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Volume filter: volume > 1.8x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(upper_channel_aligned[i]) or np.isnan(lower_channel_aligned[i]) or
            np.isnan(atr_aligned[i]) or np.isnan(ema_200[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: breakdown below lower channel or trend change
            if low[i] <= lower_channel_aligned[i] or close[i] < ema_200[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit conditions: breakout above upper channel or trend change
            if high[i] >= upper_channel_aligned[i] or close[i] > ema_200[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: breakout above upper channel with volume
            if high[i] > upper_channel_aligned[i] and close[i] > ema_200[i] and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below lower channel with volume
            elif low[i] < lower_channel_aligned[i] and close[i] < ema_200[i] and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals