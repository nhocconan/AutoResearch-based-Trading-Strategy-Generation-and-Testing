#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla Pivot Reversal with Volume Confirmation.
# Uses daily Camarilla pivot levels (R3, R4, S3, S4) calculated from prior day's OHLC.
# Fade at R3/S3 (mean reversion) and breakout at R4/S4 (trend continuation).
# Volume filter (current volume > 1.5x 20-period average) ensures quality signals.
# Works in bull/bear markets via price action at key pivot levels.
# Target: 50-150 trades over 4 years (12-37/year).

name = "6h_camarilla_pivot_rev_v1"
timeframe = "6h"
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
    
    # Daily OHLC for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    r4 = np.full(len(close_1d), np.nan)
    r3 = np.full(len(close_1d), np.nan)
    s3 = np.full(len(close_1d), np.nan)
    s4 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i])):
            rng = high_1d[i] - low_1d[i]
            r4[i] = close_1d[i] + (rng * 1.1 / 2)
            r3[i] = close_1d[i] + (rng * 1.1 / 4)
            s3[i] = close_1d[i] - (rng * 1.1 / 4)
            s4[i] = close_1d[i] - (rng * 1.1 / 2)
    
    # Align pivot levels to 6h timeframe (shifted by 1 daily bar)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if pivot data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price reaches S3 (mean reversion target) or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.0 * atr_approx
            
            if (close[i] <= s3_aligned[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches R3 (mean reversion target) or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.0 * atr_approx
            
            if (close[i] >= r3_aligned[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if volume_filter:
                # Fade at R3/S3 (mean reversion)
                # Short at R3: price rejects resistance
                if (close[i] < r3_aligned[i] and close[i-1] >= r3_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                # Long at S3: price finds support
                elif (close[i] > s3_aligned[i] and close[i-1] <= s3_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Breakout at R4/S4 (trend continuation)
                # Buy breakout above R4
                elif (close[i] > r4_aligned[i] and close[i-1] <= r4_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Sell breakdown below S4
                elif (close[i] < s4_aligned[i] and close[i-1] >= s4_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals