#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 12h trend filter and volume confirmation.
# In low volatility regimes (BB Width < 20th percentile), price is primed for breakout.
# We enter long when price breaks above upper BB with volume spike in bullish 12h trend (close > 12h EMA50).
# We enter short when price breaks below lower BB with volume spike in bearish 12h trend (close < 12h EMA50).
# This captures explosive moves after consolidation, works in both bull and bear markets by using 12h trend filter.

name = "6h_BB_Squeeze_Breakout_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Bollinger Bands (20, 2) on 6h
    bb_period = 20
    bb_std = 2
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_20 + (bb_std * std_20)
    lower_bb = sma_20 - (bb_std * std_20)
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    
    # Bollinger Band Squeeze: width < 20th percentile of last 50 periods
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=20).quantile(0.20).values
    squeeze = bb_width < bb_width_percentile
    
    # Volume confirmation: current volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(bb_width_percentile[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        bb_w = bb_width[i]
        squeeze_now = squeeze[i]
        vol_spike = volume_spike[i]
        ema_trend = ema_50_12h_aligned[i]
        upper = upper_bb[i]
        lower = lower_bb[i]
        
        # Determine 12h trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Breakout conditions
        long_breakout = close_val > upper
        short_breakout = close_val < lower
        
        # Generate signals
        if position == 0:
            # Look for breakout from squeeze with volume spike and trend alignment
            if squeeze_now and long_breakout and vol_spike and is_bull_trend:
                signals[i] = 0.25
                position = 1
            elif squeeze_now and short_breakout and vol_spike and is_bear_trend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: loss of bullish momentum or volatility expansion (end of squeeze)
            if close_val < sma_20[i] or not squeeze_now:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: loss of bearish momentum or volatility expansion
            if close_val > sma_20[i] or not squeeze_now:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals