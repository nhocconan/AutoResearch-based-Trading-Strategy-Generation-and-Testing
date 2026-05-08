#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Squeeze with 1d trend filter and volume confirmation
# Long when Bollinger Bands squeeze (bandwidth < 50th percentile) + price breaks above upper band + 1d EMA50 uptrend + volume spike
# Short when Bollinger Bands squeeze + price breaks below lower band + 1d EMA50 downtrend + volume spike
# Bollinger Squeeze identifies low volatility periods primed for breakout; 1d EMA50 filters trend direction; volume confirms breakout strength
# Designed for 4h timeframe to target 20-40 trades/year (80-160 total over 4 years)

name = "4h_BollingerSqueeze_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume EMA (20-period) for volume filter
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20)
    
    # Calculate Bollinger Bands on 4h timeframe
    bb_period = 20
    bb_std = 2.0
    
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + bb_std * std
    lower_band = sma - bb_std * std
    bandwidth = (upper_band - lower_band) / sma
    
    # Calculate bandwidth percentile (50-period lookback)
    bandwidth_percentile = pd.Series(bandwidth).rolling(window=50, min_periods=30).rank(pct=True).values
    
    # Identify squeeze: bandwidth below 50th percentile
    squeeze = bandwidth_percentile < 0.5
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(ema_50_aligned[i]) or np.isnan(vol_ema_20_aligned[i]) or \
           np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(squeeze[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 4h volume > 1.5x 20-period EMA of volume
        vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
        vol_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for entry: squeeze breakout + trend + volume
            long_condition = squeeze[i] and close[i] > upper_band[i] and close[i] > ema_50_aligned[i] and vol_filter
            short_condition = squeeze[i] and close[i] < lower_band[i] and close[i] < ema_50_aligned[i] and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below middle band or volatility expands (bandwidth > 80th percentile)
            if close[i] < sma[i] or bandwidth_percentile[i] > 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above middle band or volatility expands
            if close[i] > sma[i] or bandwidth_percentile[i] > 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals