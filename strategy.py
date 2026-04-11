#!/usr/bin/env python3
"""
1d_1w_camarilla_breakout_volume_v1
Strategy: Daily Camarilla pivot breakout with volume confirmation and weekly trend filter
Timeframe: 1d
Leverage: 1.0
Hypothesis: Uses daily Camarilla pivot levels (based on previous day's OHLC) for breakout entries, confirmed by volume spikes (>2x average) and filtered by weekly EMA200 trend direction. Designed to capture strong momentum moves while avoiding false breakouts in choppy markets. Works in bull markets (breakouts with trend) and bear markets (breakouts against trend filtered out). Target: 20-60 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels from previous day's OHLC
    # Camarilla levels: H4 = C + 1.5*(H-L), L4 = C - 1.5*(H-L)
    # where C, H, L are previous day's close, high, low
    prev_close = np.concatenate([[np.nan], close[:-1]])
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    
    # Calculate Camarilla levels
    camarilla_high = prev_close + 1.5 * (prev_high - prev_low)
    camarilla_low = prev_close - 1.5 * (prev_high - prev_low)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)  # Strong volume confirmation
    
    # Weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after enough data for indicators
        # Skip if any required data is invalid
        if (np.isnan(camarilla_high[i]) or np.isnan(camarilla_low[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below weekly EMA200
        uptrend_1w = price_close > ema_200_1w_aligned[i]
        downtrend_1w = price_close < ema_200_1w_aligned[i]
        
        # Breakout conditions
        breakout_up = price_close > camarilla_high[i]  # Break above H4
        breakout_down = price_close < camarilla_low[i]  # Break below L4
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: upward breakout with volume in uptrend
        long_signal = breakout_up and vol_confirmed and uptrend_1w
        
        # Short: downward breakout with volume in downtrend
        short_signal = breakout_down and vol_confirmed and downtrend_1w
        
        # Exit when price returns to previous day's close (pivot)
        pivot = prev_close[i]
        exit_long = position == 1 and price_close < pivot
        exit_short = position == -1 and price_close > pivot
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals