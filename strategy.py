#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d trend filter and volume confirmation
# Bollinger Band width measures volatility contraction (squeeze). Breakouts from squeezes
# often precede strong moves. Combined with 1d EMA50 trend filter to avoid counter-trend
# breakouts, and volume confirmation to ensure follow-through. Designed for 6h timeframe
# to target 12-37 trades/year (50-150 total over 4 years) with discrete sizing (0.25).
# Works in bull markets by buying upside breakouts in uptrends and in bear markets by
# selling downside breakouts in downtrends, avoiding false breakouts in ranging markets.

name = "6h_BollingerSqueeze_Breakout_1dEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Bollinger Bands (20, 2) on 6h data
    close_series = pd.Series(close)
    basis = close_series.rolling(window=20, min_periods=20).mean().values
    dev = close_series.rolling(window=20, min_periods=20).std().values
    upper = basis + 2.0 * dev
    lower = basis - 2.0 * dev
    
    # Bollinger Band Width (normalized)
    bb_width = (upper - lower) / basis
    bb_width = np.where(basis == 0, 0, bb_width)
    
    # Bollinger Band Squeeze: BB Width below 20-period rolling mean of BB Width
    bb_width_series = pd.Series(bb_width)
    bb_width_ma = bb_width_series.rolling(window=20, min_periods=20).mean().values
    squeeze_condition = bb_width < bb_width_ma
    
    # Volume confirmation: 1.5x 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(basis[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirmed = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Long: BB squeeze breakout above upper band + volume confirmation + price above 1d EMA50 (uptrend)
            if (close[i] > upper[i] and squeeze_condition[i] and volume_confirmed and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: BB squeeze breakout below lower band + volume confirmation + price below 1d EMA50 (downtrend)
            elif (close[i] < lower[i] and squeeze_condition[i] and volume_confirmed and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below basis (mean reversion) OR below 1d EMA50 (trend change)
            if close[i] < basis[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above basis (mean reversion) OR above 1d EMA50 (trend change)
            if close[i] > basis[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals