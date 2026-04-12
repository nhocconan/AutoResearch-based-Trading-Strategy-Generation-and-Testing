#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Bollinger Band squeeze breakout with 1d volume confirmation
    # - Squeeze: BB width < 20th percentile of last 50 bars → low volatility
    # - Breakout: price closes outside BB(20,2) → volatility expansion
    # - Volume: 1d volume > 1.5x 20-period average → institutional participation
    # - Direction: 1d close > 1d EMA50 for longs, < EMA50 for shorts
    # Works in bull/bear: squeeze breakouts capture volatility expansion in any regime
    # Discrete sizing 0.25 to minimize fee churn. Target: 15-25 trades/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 6h Bollinger Bands (20,2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + (bb_std * std)
    lower = sma - (bb_std * std)
    bb_width = upper - lower
    
    # Calculate BB width percentile (20th) over last 50 bars
    bb_width_percentile = np.zeros(n)
    for i in range(bb_period + 50, n):
        window_start = max(0, i - 50)
        window_end = i
        if window_end - window_start >= bb_period:
            bb_width_window = bb_width[window_start:window_end]
            if len(bb_width_window) > 0 and not np.all(np.isnan(bb_width_window)):
                valid_widths = bb_width_window[~np.isnan(bb_width_window)]
                if len(valid_widths) > 0:
                    percentile_20 = np.percentile(valid_widths, 20)
                    bb_width_percentile[i] = percentile_20
    
    # Squeeze condition: BB width < 20th percentile
    squeeze = bb_width < bb_width_percentile
    
    # Breakout conditions
    breakout_up = close > upper
    breakout_down = close < lower
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1d volume average (20-period) for volume confirmation
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    # Volume confirmation: current 1d volume > 1.5x 20-period average
    volume_confirm = volume_1d_aligned > (1.5 * vol_avg_1d_aligned)
    
    # Trend filter: 1d close > EMA50 for longs, < EMA50 for shorts
    bullish_trend = close_1d > ema50_1d
    bearish_trend = close_1d < ema50_1d
    bullish_trend_aligned = align_htf_to_ltf(prices, df_1d, bullish_trend)
    bearish_trend_aligned = align_htf_to_ltf(prices, df_1d, bearish_trend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(bb_period + 50, n):
        # Skip if data not ready
        if (np.isnan(sma[i]) or np.isnan(std[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]) or
            np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Check for squeeze breakout with volume and trend confirmation
        long_breakout = squeeze[i-1] and breakout_up[i] and volume_confirm[i] and bullish_trend_aligned[i]
        short_breakout = squeeze[i-1] and breakout_down[i] and volume_confirm[i] and bearish_trend_aligned[i]
        
        # Exit conditions: opposite breakout or loss of squeeze (volatility contraction)
        long_exit = breakout_down[i] or not squeeze[i]
        short_exit = breakout_up[i] or not squeeze[i]
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
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
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_bb_squeeze_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0