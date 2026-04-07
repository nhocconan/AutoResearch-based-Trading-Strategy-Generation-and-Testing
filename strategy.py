#!/usr/bin/env python3
"""
1d_camarilla_pivot_1w_trend_volume_v1
Hypothesis: On daily timeframe, use Camarilla pivot levels with weekly trend filter and volume confirmation.
Long when price closes above weekly EMA(50) and breaks above daily Camarilla H4 resistance with volume > 1.5x average.
Short when price closes below weekly EMA(50) and breaks below daily Camarilla L4 support with volume > 1.5x average.
Exit when price returns to the Camarilla pivot point (midpoint).
Designed for 10-25 trades/year to minimize fee drift while capturing institutional levels with weekly trend alignment.
Works in both bull and bear markets as Camarilla levels adapt to volatility and weekly trend filter avoids counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_pivot_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Determine weekly trend direction (using EMA slope)
    weekly_trend_up = np.zeros(len(ema_50_1w_aligned), dtype=bool)
    weekly_trend_down = np.zeros(len(ema_50_1w_aligned), dtype=bool)
    for i in range(1, len(ema_50_1w_aligned)):
        if not np.isnan(ema_50_1w_aligned[i]) and not np.isnan(ema_50_1w_aligned[i-1]):
            weekly_trend_up[i] = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
            weekly_trend_down[i] = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]
    
    # Calculate daily Camarilla pivot levels (using previous day's OHLC)
    # Camarilla formulas:
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.25 * (high - low)
    # H2 = close + 1.166 * (high - low)
    # H1 = close + 0.833 * (high - low)
    # Pivot = (high + low + close) / 3
    # L1 = close - 0.833 * (high - low)
    # L2 = close - 1.166 * (high - low)
    # L3 = close - 1.25 * (high - low)
    # L4 = close - 1.5 * (high - low)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    h4 = prev_close + 1.5 * range_hl
    l4 = prev_close - 1.5 * range_hl
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(h4[i]) or np.isnan(l4[i]) or 
            np.isnan(pivot[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price returns to Camarilla pivot point
            if close[i] <= pivot[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to Camarilla pivot point
            if close[i] >= pivot[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation and weekly trend alignment
            if vol_ok:
                # Long: price closes above weekly EMA(50) and breaks above H4
                if (close[i] > ema_50_1w_aligned[i] and 
                    close[i] > h4[i] and close[i-1] <= h4[i-1]):
                    position = 1
                    signals[i] = 0.25
                # Short: price closes below weekly EMA(50) and breaks below L4
                elif (close[i] < ema_50_1w_aligned[i] and 
                      close[i] < l4[i] and close[i-1] >= l4[i-1]):
                    position = -1
                    signals[i] = -0.25
    
    return signals