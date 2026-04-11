#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with weekly candlestick pattern confirmation and daily volume.
# Uses weekly bullish/bearish engulfing patterns for entry, with weekly trend filter to avoid counter-trend trades.
# Volume filter ensures institutional participation. Designed for 12-37 trades/year on 12h.
# Weekly trend filter reduces whipsaw in sideways markets and improves win rate in both bull and bear markets.

name = "12h_1w_engulfing_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Previous weekly values for pattern detection
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    prev_open_1w = np.roll(open_1w, 1)
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    prev_close_1w[0] = np.nan
    prev_open_1w[0] = np.nan
    
    # Weekly bullish engulfing: current green candle fully engulfs previous red candle
    bullish_engulf = (close_1w > open_1w) & (open_1w <= prev_close_1w) & (close_1w >= prev_open_1w) & (prev_close_1w < prev_open_1w)
    # Weekly bearish engulfing: current red candle fully engulfs previous green candle
    bearish_engulf = (close_1w < open_1w) & (open_1w >= prev_close_1w) & (close_1w <= prev_open_1w) & (prev_close_1w > prev_open_1w)
    
    # Weekly trend: price above/below weekly 20 EMA
    close_series_1w = pd.Series(close_1w)
    ema20_1w = close_series_1w.ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_trend_bull = close_1w > ema20_1w
    weekly_trend_bear = close_1w < ema20_1w
    
    # Align weekly signals to 12h
    bullish_engulf_aligned = align_htf_to_ltf(prices, df_1w, bullish_engulf)
    bearish_engulf_aligned = align_htf_to_ltf(prices, df_1w, bearish_engulf)
    weekly_trend_bull_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_bull)
    weekly_trend_bear_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_bear)
    
    # Daily average volume (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_avg_20 = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_avg_20[i] = np.mean(volume_1d[i-19:i+1])
    
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(bullish_engulf_aligned[i]) or np.isnan(bearish_engulf_aligned[i]) or
            np.isnan(vol_avg_aligned[i]) or
            np.isnan(weekly_trend_bull_aligned[i]) or np.isnan(weekly_trend_bear_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * daily average volume
        vol_filter = volume[i] > 1.5 * vol_avg_aligned[i]
        
        # Determine weekly trend direction
        is_bullish_week = weekly_trend_bull_aligned[i]
        is_bearish_week = weekly_trend_bear_aligned[i]
        
        # Enter long on weekly bullish engulfing in bullish weekly trend
        enter_long = bullish_engulf_aligned[i] and vol_filter and is_bullish_week
        # Enter short on weekly bearish engulfing in bearish weekly trend
        enter_short = bearish_engulf_aligned[i] and vol_filter and is_bearish_week
        
        # Exit when opposite engulfing pattern forms or trend changes
        exit_long = (position == 1 and 
                    (bearish_engulf_aligned[i] or not is_bullish_week))
        exit_short = (position == -1 and 
                     (bullish_engulf_aligned[i] or not is_bearish_week))
        
        # Priority: entry > exit > hold
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals