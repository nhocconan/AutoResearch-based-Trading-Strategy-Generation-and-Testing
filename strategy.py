#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d volume spike and 1w trend filter.
# Bollinger Band squeeze (low volatility) precedes explosive moves. Breakout direction
# confirmed by 1d volume spike (>2x 20-period average). 1w EMA50 filter ensures we
# only trade breakouts in the direction of the weekly trend to avoid false breakouts.
# Works in both bull and bear markets by capturing volatility expansion phases.
# Target: 12-37 trades/year (50-150 over 4 years).

name = "6h_BBSqueeze_1dVolSpike_1wTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume 20-period EMA for spike detection
    vol_ema20_1d = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_1d = df_1d['volume'].values > (2.0 * vol_ema20_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 6h Bollinger Bands (20, 2)
    sma20_6h = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20_6h = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb_6h = sma20_6h + (2.0 * std20_6h)
    lower_bb_6h = sma20_6h - (2.0 * std20_6h)
    bb_width_6h = (upper_bb_6h - lower_bb_6h) / sma20_6h
    
    # Bollinger Band squeeze: bb_width below 20-period rolling 10th percentile
    bb_width_series = pd.Series(bb_width_6h)
    bb_width_percentile_10 = bb_width_series.rolling(window=20, min_periods=20).quantile(0.10).values
    squeeze_condition = bb_width_6h < bb_width_percentile_10
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    squeeze_active = False  # Track if we are in a squeeze state
    
    for i in range(30, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(close[i]) or np.isnan(sma20_6h[i]) or np.isnan(std20_6h[i]) or 
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Update squeeze state
        if squeeze_condition[i]:
            squeeze_active = True
        else:
            # Only exit squeeze state when width expands significantly
            if bb_width_6h[i] > (bb_width_percentile_10[i] * 1.5):
                squeeze_active = False
        
        if position == 0:
            # Look for breakout after squeeze
            if squeeze_active:
                # Long breakout: price closes above upper BB with volume spike and above weekly EMA50
                if (close[i] > upper_bb_6h[i] and 
                    volume_spike_1d_aligned[i] > 0.5 and  # At least some volume confirmation
                    close[i] > ema50_1w_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    squeeze_active = False  # Reset squeeze after breakout
                # Short breakout: price closes below lower BB with volume spike and below weekly EMA50
                elif (close[i] < lower_bb_6h[i] and 
                      volume_spike_1d_aligned[i] > 0.5 and
                      close[i] < ema50_1w_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    squeeze_active = False  # Reset squeeze after breakout
        elif position == 1:
            # Exit long: price re-enters Bollinger Bands or reverse squeeze breakout
            if close[i] < sma20_6h[i] or (close[i] < lower_bb_6h[i] and volume_spike_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Bollinger Bands or reverse squeeze breakout
            if close[i] > sma20_6h[i] or (close[i] > upper_bb_6h[i] and volume_spike_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals