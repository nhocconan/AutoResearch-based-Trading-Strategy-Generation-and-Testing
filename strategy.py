#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Weekly Opening Gap Reversal with Volume Confirmation.
# Fades price gaps that occur between weekly candles (Sunday open vs Friday close).
# Uses weekly open/gap calculation: gap = (weekly_open - prev_weekly_close) / prev_weekly_close.
# Enters when price fills 50% of the gap with volume confirmation (>1.5x 20-period average).
# Works in bull/bear markets as gap fills are mean-reverting tendencies.
# Target: 50-150 trades over 4 years (12-37/year).

name = "12h_weekly_gap_fill_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for gap calculation
    df_1w = get_htf_data(prices, '1w')
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    
    # Calculate weekly gap percentage: (weekly_open - prev_weekly_close) / prev_weekly_close
    gap_pct = np.full(len(weekly_open), np.nan)
    for i in range(1, len(weekly_open)):
        if not (np.isnan(weekly_open[i]) or np.isnan(weekly_close[i-1]) or weekly_close[i-1] == 0):
            gap_pct[i] = (weekly_open[i] - weekly_close[i-1]) / weekly_close[i-1]
    
    # Align gap percentage to 12h timeframe (shifted by 1 weekly bar)
    gap_aligned = align_htf_to_ltf(prices, df_1w, gap_pct)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if gap data not available
        if np.isnan(gap_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price fills the gap (returns to weekly open) or stoploss
            weekly_open_price = weekly_open[np.searchsorted(df_1w.index.values[:len(weekly_open)], 
                                                         prices.index[i]) - 1] if i >= 12 else weekly_open[0]
            if np.isnan(weekly_open_price):
                weekly_open_price = weekly_open[0]
            
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.0 * atr_approx
            
            if (close[i] >= weekly_open_price or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price fills the gap (returns to weekly open) or stoploss
            weekly_open_price = weekly_open[np.searchsorted(df_1w.index.values[:len(weekly_open)], 
                                                         prices.index[i]) - 1] if i >= 12 else weekly_open[0]
            if np.isnan(weekly_open_price):
                weekly_open_price = weekly_open[0]
            
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.0 * atr_approx
            
            if (close[i] <= weekly_open_price or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if volume_filter and abs(gap_aligned[i]) > 0.005:  # Minimum 0.5% gap
                # For negative gap (weekly open < prev weekly close), look for long when price fills 50% of gap
                if gap_aligned[i] < 0:
                    weekly_close_price = weekly_close[np.searchsorted(df_1w.index.values[:len(weekly_close)], 
                                                                     prices.index[i]) - 1] if i >= 12 else weekly_close[0]
                    weekly_open_price = weekly_open[np.searchsorted(df_1w.index.values[:len(weekly_open)], 
                                                                   prices.index[i]) - 1] if i >= 12 else weekly_open[0]
                    if not (np.isnan(weekly_close_price) or np.isnan(weekly_open_price)):
                        gap_fill_50 = weekly_close_price + (gap_aligned[i] * weekly_close_price * 0.5)
                        if (close[i] >= gap_fill_50 and close[i-1] < gap_fill_50):
                            signals[i] = 0.25
                            position = 1
                            entry_price = close[i]
                # For positive gap (weekly open > prev weekly close), look for short when price fills 50% of gap
                elif gap_aligned[i] > 0:
                    weekly_close_price = weekly_close[np.searchsorted(df_1w.index.values[:len(weekly_close)], 
                                                                     prices.index[i]) - 1] if i >= 12 else weekly_close[0]
                    weekly_open_price = weekly_open[np.searchsorted(df_1w.index.values[:len(weekly_open)], 
                                                                   prices.index[i]) - 1] if i >= 12 else weekly_open[0]
                    if not (np.isnan(weekly_close_price) or np.isnan(weekly_open_price)):
                        gap_fill_50 = weekly_close_price + (gap_aligned[i] * weekly_close_price * 0.5)
                        if (close[i] <= gap_fill_50 and close[i-1] > gap_fill_50):
                            signals[i] = -0.25
                            position = -1
                            entry_price = close[i]
    
    return signals