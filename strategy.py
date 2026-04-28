#102439
#!/usr/bin/env python3
"""
6h_ADX_Keltner_Trend_Filter_12hTrend_Filter_Volume
Hypothesis: Combines ADX trend strength with Keltner Channel breakouts, filtered by 12h trend direction and volume spike. ADX > 25 indicates strong trend, price breaking above/below Keltner Channel (EMA20 ± 2*ATR) signals entry in trend direction. Volume > 1.5x average confirms momentum. Designed to work in both bull and bear by following trend direction. Targets 20-30 trades/year via strict ADX and channel breakout conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA20 for trend filter
    close_12h = df_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Calculate ADX (14)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        alpha = 1.0 / period
        result[period-1] = np.nansum(data[:period]) if not np.isnan(data[:period]).any() else np.nan
        for i in range(period, len(data)):
            if not np.isnan(data[i]) and not np.isnan(result[i-1]):
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
            else:
                result[i] = np.nan
        return result
    
    # Calculate smoothed values
    atr_raw = wilder_smooth(tr, 14)
    plus_di_raw = wilder_smooth(plus_dm, 14)
    minus_di_raw = wilder_smooth(minus_dm, 14)
    
    # Avoid division by zero
    dx = np.zeros(n)
    for i in range(n):
        if not np.isnan(plus_di_raw[i]) and not np.isnan(minus_di_raw[i]) and (plus_di_raw[i] + minus_di_raw[i]) != 0:
            dx[i] = abs(plus_di_raw[i] - minus_di_raw[i]) / (plus_di_raw[i] + minus_di_raw[i]) * 100
        else:
            dx[i] = np.nan
    
    adx = wilder_smooth(dx, 14)
    
    # Calculate Keltner Channel (EMA20 ± 2*ATR)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr = atr_raw * 2  # 2*ATR for channel width
    upper_keltner = ema_20 + atr
    lower_keltner = ema_20 - atr
    
    # Volume confirmation: >1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for ADX to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx[i]) or 
            np.isnan(ema_20_12h_aligned[i]) or
            np.isnan(upper_keltner[i]) or
            np.isnan(lower_keltner[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA20
        uptrend_12h = close[i] > ema_20_12h_aligned[i]
        downtrend_12h = close[i] < ema_20_12h_aligned[i]
        
        # ADX trend strength (>25 = strong trend)
        strong_trend = adx[i] > 25
        
        # Volume confirmation (>1.5x average)
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Keltner Channel breakout
        breakout_up = close[i] > upper_keltner[i]
        breakout_down = close[i] < lower_keltner[i]
        
        # Entry conditions
        long_entry = breakout_up and strong_trend and vol_confirm and uptrend_12h
        short_entry = breakout_down and strong_trend and vol_confirm and downtrend_12h
        
        # Exit conditions: return to EMA20 (middle of channel)
        long_exit = close[i] < ema_20[i]
        short_exit = close[i] > ema_20[i]
        
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

name = "6h_ADX_Keltner_Trend_Filter_12hTrend_Filter_Volume"
timeframe = "6h"
leverage = 1.0