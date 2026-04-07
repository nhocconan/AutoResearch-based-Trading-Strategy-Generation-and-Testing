#!/usr/bin/env python3
"""
12h_camarilla_pivot_1w_ema_volume_v1
Hypothesis: Uses Camarilla pivot levels on 12h with 1-week EMA200 trend filter. Long when price touches S3 with strong volume and price above weekly EMA200, short when price touches R3 with strong volume and price below weekly EMA200. Camarilla levels provide high-probability reversal zones, weekly EMA filters trend direction, and volume confirmation avoids false signals. Designed for low-frequency, high-quality trades in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1w_ema_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-week EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate Camarilla pivot levels for each 12h bar using previous day's OHLC
    # We need daily OHLC to calculate pivots for 12h bars
    # Create daily OHLC from 12h data by resampling conceptually (but using actual logic)
    # For each 12h bar, we use the previous day's daily high, low, close
    
    # First, create arrays for daily OHLC (we'll compute this by grouping 12h bars into days)
    # Since we have 12h data, two 12h bars make one day
    # We'll shift the daily OHLC by 1 to avoid look-ahead
    
    # Calculate daily high, low, close from 12h data (two 12h bars = 1 day)
    # We'll use rolling window of 2 to get daily values, then shift by 2 to get previous day
    if n >= 2:
        # Two 12h bars make one day
        daily_high = np.maximum(high[:-1:2], high[1::2]) if len(high) >= 2 else high
        daily_low = np.minimum(low[:-1:2], low[1::2]) if len(low) >= 2 else low
        daily_close = close[1::2] if len(close) >= 2 else close
        
        # Pad to match original length (approximate)
        # For simplicity, we'll use previous available daily data
        # In practice, we calculate pivots using available daily OHLC
        
        # Simpler approach: use typical price for pivot calculation with lookback
        typical_price = (high + low + close) / 3
        
        # Use 2-period lookback for previous day's typical price (since 2x12h = 1 day)
        prev_typical = np.roll(typical_price, 2)
        prev_typical[:2] = typical_price[0]  # fill beginning
        
        # But better: calculate proper Camarilla using session OHLC
        # For 12h chart, we can use previous 12h bar's range as proxy
        # Actually, Camarilla uses previous day's OHLC
        
        # Let's use a rolling window approach with proper shift
        # We'll calculate using the previous day's data by looking back 2 bars
        
        # Initialize pivot arrays
        s3 = np.full(n, np.nan)
        r3 = np.full(n, np.nan)
        
        # Start from index 2 to ensure we have previous day's data
        for i in range(2, n):
            # Get previous day's OHLC - since we have 12h bars, 
            # we need to aggregate the two 12h bars from previous day
            # Simpler: use 2-bar lookback for daily equivalent
            
            # For bar i, previous day's data is bars i-2 and i-1
            if i >= 2:
                # Previous day's high, low, close
                prev_high = np.max(high[i-2:i]) if i >= 2 else high[0]
                prev_low = np.min(low[i-2:i]) if i >= 2 else low[0]
                prev_close = close[i-1] if i >= 1 else close[0]
                
                # Calculate Camarilla levels
                range_val = prev_high - prev_low
                if range_val > 0:
                    s3 = prev_close - 1.1 * range_val * 1.1666  # S3 level
                    r3 = prev_close + 1.1 * range_val * 1.1666  # R3 level
                    
                    s3[i] = s3
                    r3[i] = r3
                else:
                    s3[i] = s3[i-1] if i > 0 else close[i]
                    r3[i] = r3[i-1] if i > 0 else close[i]
            else:
                s3[i] = close[i]
                r3[i] = close[i]
    else:
        s3 = np.full(n, close[0])
        r3 = np.full(n, close[0])
    
    # Volume confirmation: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(s3[i]) or np.isnan(r3[i]) or 
            np.isnan(close[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price moves back above S3 or trend changes
            if close[i] > s3[i] or close[i] < ema_200_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves back below R3 or trend changes
            if close[i] < r3[i] or close[i] > ema_200_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price touches S3 (or goes below) with volume confirmation and uptrend
            if close[i] <= s3[i] and vol_confirmed and close[i] > ema_200_1w_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: price touches R3 (or goes above) with volume confirmation and downtrend
            elif close[i] >= r3[i] and vol_confirmed and close[i] < ema_200_1w_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals