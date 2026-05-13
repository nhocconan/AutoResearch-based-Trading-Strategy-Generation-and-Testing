#!/usr/bin/env python3
"""
1d_WeeklyTrend_RSIDivergence_v1
Hypothesis: For 1d timeframe, use weekly trend filter (EMA34) combined with daily RSI divergence signals. In bull markets (price > weekly EMA34), look for bullish RSI divergence (price making lower low, RSI making higher low) to go long. In bear markets (price < weekly EMA34), look for bearish RSI divergence (price making higher high, RSI making lower high) to go short. Uses volume confirmation to filter false signals. Designed for low trade frequency (5-15/year) with strong directional moves, minimizing fee impact while capturing major trend reversals.
"""

name = "1d_WeeklyTrend_RSIDivergence_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def rsi(close, period=14):
    """Calculate RSI with proper Wilder's smoothing"""
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], 100)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def find_divergence(close, rsi_vals, lookback=10):
    """
    Find bullish/bearish divergence over lookback period
    Returns: 1 for bullish div, -1 for bearish div, 0 otherwise
    """
    n = len(close)
    bullish_div = np.zeros(n)
    bearish_div = np.zeros(n)
    
    for i in range(lookback, n):
        # Get lookback window
        window_close = close[i-lookback:i+1]
        window_rsi = rsi_vals[i-lookback:i+1]
        
        # Find local minima and maxima in window
        min_idx = np.argmin(window_close)
        max_idx = np.argmax(window_close)
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        if min_idx == lookback:  # Current point is lowest in window
            if i > lookback:  # Need previous point to compare
                # Find previous low in window (excluding current)
                prev_window_close = close[i-lookback:i]
                prev_min_idx = np.argmin(prev_window_close)
                if prev_min_idx == len(prev_window_close) - 1:  # Previous point was lowest
                    if window_close[-1] < window_close[-2] and window_rsi[-1] > window_rsi[-2]:
                        bullish_div[i] = 1
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        if max_idx == lookback:  # Current point is highest in window
            if i > lookback:
                prev_window_close = close[i-lookback:i]
                prev_max_idx = np.argmax(prev_window_close)
                if prev_max_idx == len(prev_window_close) - 1:
                    if window_close[-1] > window_close[-2] and window_rsi[-1] < window_rsi[-2]:
                        bearish_div[i] = -1
    
    return bullish_div + bearish_div  # Returns 1 for bullish, -1 for bearish

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate daily RSI
    rsi_vals = rsi(close, 14)
    
    # Find RSI divergences
    divergence_signal = find_divergence(close, rsi_vals, lookback=8)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirmed = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(35, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(rsi_vals[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine market regime: bullish if price > weekly EMA34, bearish if price < weekly EMA34
        is_bullish = close[i] > ema_34_1w_aligned[i]
        is_bearish = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # LONG: Bullish market + bullish RSI divergence + volume confirmation
            if is_bullish and divergence_signal[i] == 1 and vol_confirmed[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish market + bearish RSI divergence + volume confirmation
            elif is_bearish and divergence_signal[i] == -1 and vol_confirmed[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish divergence appears or trend turns bearish
            if divergence_signal[i] == -1 or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish divergence appears or trend turns bullish
            if divergence_signal[i] == 1 or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals