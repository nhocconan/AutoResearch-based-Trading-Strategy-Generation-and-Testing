#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour momentum with 4-hour trend filter and session filter (08-20 UTC).
# Long when 1h price crosses above 12-period EMA + 4h EMA(21) upward + volume > 1.5x 20-period average + session active.
# Short when 1h price crosses below 12-period EMA + 4h EMA(21) downward + volume > 1.5x 20-period average + session active.
# Exit when price crosses 8-period EMA in opposite direction.
# Stoploss at 2.0 * ATR(14).
# Position size: 0.20 (20% of capital).
# Uses 4h EMA for trend direction and 1h volume for confirmation.
# Target: 60-150 total trades over 4 years (15-37/year).

name = "1h_ema12_4h_trend_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4-hour data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # Calculate 4-hour EMA(21) for trend
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1-hour EMA(12) for entry
    ema_12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # Calculate 1-hour EMA(8) for exit
    ema_8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    # Calculate 1-hour volume average (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(21, n):
        # Skip if required data not available
        if (np.isnan(ema_12[i]) or np.isnan(ema_8[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below 8-period EMA
            elif close[i] < ema_8[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above 8-period EMA
            elif close[i] > ema_8[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: EMA crossover with 4h trend filter, volume confirmation, and session
            # EMA crossover: price crosses 12-period EMA
            ema_cross_up = close[i] > ema_12[i] and close[i-1] <= ema_12[i-1]
            ema_cross_down = close[i] < ema_12[i] and close[i-1] >= ema_12[i-1]
            # Volume filter: volume > 1.5x 20-period average
            volume_filter = volume[i] > 1.5 * volume_ma[i]
            # Trend filter: 4h EMA(21) slope (using 3-bar change for direction)
            ema_4h_slope = ema_4h_aligned[i] - ema_4h_aligned[i-3]
            trend_up = ema_4h_slope > 0
            trend_down = ema_4h_slope < 0
            # Session filter
            session_filter = in_session[i]
            
            # Long: price crosses above EMA12 + 4h trend up + volume filter + session
            if ema_cross_up and trend_up and volume_filter and session_filter:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price crosses below EMA12 + 4h trend down + volume filter + session
            elif ema_cross_down and trend_down and volume_filter and session_filter:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals