# This strategy implements a 1-day momentum breakout system with weekly trend filter
# It uses Donchian breakouts for entry, weekly EMA for trend direction, and volume confirmation
# Designed for low trade frequency (~10-25 trades/year) to minimize fee drag in both bull and bear markets
# Works in bull markets by capturing breakouts, in bear markets by filtering counter-trend signals

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
    
    # Get weekly data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA34 for trend direction
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate daily Donchian channels (20-period)
    # Highest high and lowest low over past 20 days
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily ATR (14-period) for volatility filter and position sizing
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-day average volume for volume confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for enough data for weekly EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_1w_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(atr[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA34
        uptrend = close[i] > ema34_1w_aligned[i]
        downtrend = close[i] < ema34_1w_aligned[i]
        
        # Volatility filter: ATR > 0 (always true but keeps structure)
        vol_filter = atr[i] > 0
        
        # Volume filter: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * avg_volume[i]
        
        trade_allowed = vol_filter and vol_confirm
        
        if position == 0:
            # Long breakout: price breaks above 20-day high in uptrend
            if trade_allowed and uptrend and close[i] > highest_high[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below 20-day low in downtrend
            elif trade_allowed and downtrend and close[i] < lowest_low[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 20-day low or trend reverses
            if close[i] < lowest_low[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 20-day high or trend reverses
            if close[i] > highest_high[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_WeeklyEMA34_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0