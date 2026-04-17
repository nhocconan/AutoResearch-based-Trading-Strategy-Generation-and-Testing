#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d daily open/close gap fill strategy with 1-week EMA trend filter and volume confirmation.
# Exploits mean reversion of overnight gaps in crypto markets, which tend to fill during Asian/European sessions.
# Uses weekly EMA for trend context to avoid counter-trend trades, volume to confirm participation.
# Target: 15-25 trades/year to stay within optimal range for 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for gap calculation and 1w data for trend filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d daily gap: (today's open - yesterday's close) / yesterday's close
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    gap = (open_1d - np.roll(close_1d, 1)) / np.roll(close_1d, 1)
    gap[0] = 0  # First day has no previous close
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d gap and 1w EMA to 1d (no alignment needed for same timeframe)
    gap_aligned = gap  # Already at 1d frequency
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: current volume > 1.5 * 20-day average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need 50-period EMA + 20-period volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(gap_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 1.5x average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below 1w EMA50
        price_above_ema = close[i] > ema50_1w_aligned[i]
        price_below_ema = close[i] < ema50_1w_aligned[i]
        
        # Gap mean reversion: fade gaps > 0.5% with volume and trend alignment
        gap_threshold = 0.005  # 0.5%
        
        if position == 0:
            # Long: Gap down > -0.5%, price above weekly EMA, volume confirmation
            if (gap_aligned[i] < -gap_threshold and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Gap up > +0.5%, price below weekly EMA, volume confirmation
            elif (gap_aligned[i] > gap_threshold and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Gap fills (price returns to previous close) OR gap reverses
            if (close[i] >= close_1d[i-1]) or (gap_aligned[i] > -gap_threshold/2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Gap fills (price returns to previous close) OR gap reverses
            if (close[i] <= close_1d[i-1]) or (gap_aligned[i] < gap_threshold/2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_GapFill_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0