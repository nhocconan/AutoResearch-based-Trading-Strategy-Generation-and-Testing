#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_Volume_Trend
1d strategy using daily Camarilla pivot levels with volume confirmation and weekly trend filter.
- Long: Close > H3 + volume > 1.5x weekly avg + weekly EMA34 > EMA89
- Short: Close < L3 + volume > 1.5x weekly avg + weekly EMA34 < EMA89
- Exit: Opposite pivot level touch or trend reversal
Designed for ~5-15 trades/year per symbol (20-60 total over 4 years)
Works in bull markets (trend continuation) and bear markets (mean reversion at extremes)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter and volume average
    df_1w = get_htf_data(prices, '1w')
    
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly EMA34 and EMA89 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_1w = pd.Series(close_1w).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    ema_89_aligned = align_htf_to_ltf(prices, df_1w, ema_89_1w)
    
    # Weekly volume average (10-period)
    vol_ma_10 = pd.Series(volume_1w).rolling(window=10, min_periods=10).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_10)
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # H3 = close + 1.1*(high-low)*1.1/2
    # L3 = close - 1.1*(high-low)*1.1/2
    # Using previous day's OHLC for current day's levels
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # first bar uses current bar
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to daily timeframe (no additional delay needed)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 89  # need enough for EMA89
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_aligned[i]) or np.isnan(ema_89_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_34_aligned[i] > ema_89_aligned[i]
        downtrend = ema_34_aligned[i] < ema_89_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Camarilla pivot conditions
        above_h3 = close[i] > camarilla_h3_aligned[i]
        below_l3 = close[i] < camarilla_l3_aligned[i]
        
        if position == 0:
            # Long: uptrend + volume + above H3
            if uptrend and vol_confirm and above_h3:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + below L3
            elif downtrend and vol_confirm and below_l3:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change, volume confirmation, or below L3 (mean reversion)
            if not uptrend or (vol_confirm and below_l3):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change, volume confirmation, or above H3 (mean reversion)
            if not downtrend or (vol_confirm and above_h3):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_Pivot_Volume_Trend"
timeframe = "1d"
leverage = 1.0