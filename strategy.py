#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Weekly ATR Breakout with Volume and Trend Filter
# Hypothesis: Weekly ATR-based breakouts capture momentum in both bull and bear markets.
# Uses 1d price above/below 50 EMA for trend filter to avoid counter-trend trades.
# Volume filter ensures institutional participation. ATR stoploss manages risk.
# Target: 15-25 trades/year (60-100 over 4 years).

name = "1d_weekly_atr_breakout_volume_trend_v1"
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
    
    # Get weekly data for ATR calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 14:
        return np.zeros(n)
    
    # Calculate weekly ATR(14)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # True Range
    tr1 = weekly_high - weekly_low
    tr2 = np.abs(weekly_high - np.roll(weekly_close, 1))
    tr3 = np.abs(weekly_low - np.roll(weekly_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR using Wilder's smoothing (equivalent to RMA)
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Shift by 1 to use previous week's ATR (avoid look-ahead)
    atr_prev = np.roll(atr, 1)
    atr_prev[0] = atr[0]  # First period
    
    # Align to 1d timeframe
    atr_aligned = align_htf_to_ltf(prices, df_weekly, atr_prev)
    
    # Calculate weekly high/low for breakout levels
    weekly_high_shift = np.roll(weekly_high, 1)
    weekly_low_shift = np.roll(weekly_low, 1)
    weekly_high_shift[0] = weekly_high[0]
    weekly_low_shift[0] = weekly_low[0]
    
    weekly_high_aligned = align_htf_to_ltf(prices, df_weekly, weekly_high_shift)
    weekly_low_aligned = align_htf_to_ltf(prices, df_weekly, weekly_low_shift)
    
    # Breakout levels: weekly high/low ± 0.5 * ATR
    breakout_high = weekly_high_aligned + 0.5 * atr_aligned
    breakout_low = weekly_low_aligned - 0.5 * atr_aligned
    
    # 1d trend filter: price above/below 50 EMA
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(breakout_high[i]) or np.isnan(breakout_low[i]) or 
            np.isnan(ema_50[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below breakout_low or trend turns bearish or volume drops
            if (close[i] < breakout_low[i] or close[i] < ema_50[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above breakout_high or trend turns bullish or volume drops
            if (close[i] > breakout_high[i] or close[i] > ema_50[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above breakout_high with volume and bullish trend
            if (high[i] > breakout_high[i] and close[i] > ema_50[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below breakout_low with volume and bearish trend
            elif (low[i] < breakout_low[i] and close[i] < ema_50[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals