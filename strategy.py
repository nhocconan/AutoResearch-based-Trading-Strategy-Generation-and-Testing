#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Bands squeeze breakout with volume confirmation and 12h trend filter.
# Strategy enters long when price breaks above upper BB after low volatility (BB width < 20th percentile)
# with volume spike and 12h uptrend (price > EMA50). Short when price breaks below lower BB with
# volume spike and 12h downtrend. Exits on opposite BB touch or trend reversal.
# Designed to capture volatility expansion phases in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "6h_Bollinger_Squeeze_Breakout_12hTrend_Volume"
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
    
    # Bollinger Bands (20, 2) on 6h
    close_s = pd.Series(close)
    bb_mid = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Bollinger width percentile (20-period lookback) for squeeze detection
    bb_width_s = pd.Series(bb_width)
    bb_width_percentile = bb_width_s.rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h close for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume spike filter: current volume > 2.0 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need enough data for BB and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or 
            np.isnan(bb_width_percentile[i]) or
            np.isnan(ema50_12h_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bb_width_pct = bb_width_percentile[i]
        bb_upper_val = bb_upper[i]
        bb_lower_val = bb_lower[i]
        ema50_12h_val = ema50_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: BB squeeze breakout up + volume spike + 12h uptrend
            if (bb_width_pct < 20 and  # Bollinger squeeze
                close[i] > bb_upper_val and
                close[i] > ema50_12h_val and
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: BB squeeze breakout down + volume spike + 12h downtrend
            elif (bb_width_pct < 20 and  # Bollinger squeeze
                  close[i] < bb_lower_val and
                  close[i] < ema50_12h_val and
                  vol_spike):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price touches lower BB or 12h trend turns down
            if close[i] < bb_lower_val or close[i] < ema50_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price touches upper BB or 12h trend turns up
            if close[i] > bb_upper_val or close[i] > ema50_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals