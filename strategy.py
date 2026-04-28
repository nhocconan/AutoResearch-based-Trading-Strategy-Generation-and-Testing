# 4h_MultiTimeframe_VolatilityBreakout_Signal
# Hypothesis: 4h volatility breakout with multi-timeframe trend confirmation and volume filter.
# Uses ATR-based breakout from 4h range with 1d trend filter (EMA50) and volume spike (>1.5x 20-bar MA).
# Designed for low trade frequency (15-25 trades/year) to minimize fee decay while capturing explosive moves.
# Works in bull markets via breakouts and in bear markets via short breakdowns with trend alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h ATR(14) for volatility breakout
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h range (high-low) for breakout levels
    range_hl = high - low
    
    # Calculate volume confirmation: >1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volatility breakout conditions
        breakout_up = close[i] > close[i-1] + 0.5 * atr[i-1]  # Break above prior close + 0.5*ATR
        breakdown_down = close[i] < close[i-1] - 0.5 * atr[i-1]  # Break below prior close - 0.5*ATR
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Entry logic: volatility breakout in direction of trend with volume
        long_entry = vol_confirm and uptrend and breakout_up
        short_entry = vol_confirm and downtrend and breakdown_down
        
        # Exit logic: opposite breakout or trend change
        long_exit = breakdown_down or (not uptrend)
        short_exit = breakout_up or (not downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_MultiTimeframe_VolatilityBreakout_Signal"
timeframe = "4h"
leverage = 1.0