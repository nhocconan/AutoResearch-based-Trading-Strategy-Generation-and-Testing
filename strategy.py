#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze with 1d Trend Filter and Volume Confirmation
# Bollinger Band squeeze identifies low volatility periods that precede breakouts.
# We enter when price breaks out of the Bollinger Bands during a squeeze, in the direction of the 1d trend.
# Volume confirmation ensures the breakout is genuine. This strategy avoids whipsaws in ranging markets
# by only trading during volatility expansion phases. Targets 20-40 trades per year (~80-160 total over 4 years).

name = "6h_BollingerSqueeze_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands: 20-period SMA, 2 standard deviations
    close_series = pd.Series(close)
    sma20 = close_series.rolling(window=20, min_periods=20).mean().values
    std20 = close_series.rolling(window=20, min_periods=20).std().values
    upper = sma20 + (2 * std20)
    lower = sma20 - (2 * std20)
    
    # Bollinger Band Width for squeeze detection: (upper - lower) / sma20
    bb_width = (upper - lower) / sma20
    # Squeeze threshold: BB width below its 50-period mean (low volatility)
    bb_width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    squeeze = bb_width < bb_width_ma
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for BB and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(sma20[i]) or np.isnan(std20[i]) or np.isnan(bb_width[i]) or 
            np.isnan(bb_width_ma[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        sma20_val = sma20[i]
        upper_val = upper[i]
        lower_val = lower[i]
        squeeze_val = squeeze[i]
        ema50_val = ema50_1d_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: price breaks above upper BB during squeeze, price > 1d EMA50 (uptrend), volume confirmation
            if close_val > upper_val and squeeze_val and close_val > ema50_val and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower BB during squeeze, price < 1d EMA50 (downtrend), volume confirmation
            elif close_val < lower_val and squeeze_val and close_val < ema50_val and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle band or squeeze ends
            if close_val < sma20_val or not squeeze_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle band or squeeze ends
            if close_val > sma20_val or not squeeze_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals