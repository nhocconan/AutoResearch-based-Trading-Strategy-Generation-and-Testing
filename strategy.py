#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: 4h Camarilla R1/S1 breakout in direction of 1d EMA34 trend, confirmed by volume spike (>1.8x 20-bar MA). Uses ATR(14) trailing stop (2.0 ATR from extreme) for risk control. Designed for lower frequency (target 20-40 trades/year) with discrete sizing (0.25) to minimize fee drag and improve generalization across bull/bear markets.
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    # ATR(14) for stoploss calculation
    atr_period = 14
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(n)
    if n >= atr_period:
        atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, n):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size
    
    # Track extreme for trailing stop
    long_high = 0.0
    low_low = 0.0
    
    # Warmup: max of calculations
    start_idx = max(20, 34, 20, atr_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Determine 1d trend: bullish if price > EMA34, bearish if price < EMA34
        bullish_1d = close_val > ema_34_val
        bearish_1d = close_val < ema_34_val
        
        # Typical price for Camarilla calculation (using previous day's typical price)
        # Since we don't have daily OHLC directly, approximate using rolling window
        if i >= 24:  # Approximate 1 day back in 4h bars (24 * 4h = 96h ≈ 4 days, adjust)
            # Use previous day's typical price: (H+L+C)/3 from 1d data
            # We'll approximate using 24-bar lookback for typical price
            tp_lookback = min(24, i)
            if tp_lookback >= 1:
                prev_high = np.max(high[i-tp_lookback:i])
                prev_low = np.min(low[i-tp_lookback:i])
                prev_close = close[i-1]
                typical_price = (prev_high + prev_low + prev_close) / 3.0
                range_val = prev_high - prev_low
                
                # Camarilla levels
                R1 = typical_price + (range_val * 1.1 / 12)
                S1 = typical_price - (range_val * 1.1 / 12)
            else:
                R1 = high_val
                S1 = low_val
        else:
            R1 = high_val
            S1 = low_val
        
        # Entry conditions
        long_entry = (close_val > R1) and bullish_1d and vol_spike
        short_entry = (close_val < S1) and bearish_1d and vol_spike
        
        # Update trailing extremes
        if position == 1:
            long_high = max(long_high, high_val)
        elif position == -1:
            low_low = min(low_low, low_val)
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # ATR trailing stop or opposite breakout
            if long_high > 0 and close_val < (long_high - 2.0 * atr_val):
                exit_long = True
            elif close_val < S1:  # Opposite breakout
                exit_long = True
        elif position == -1:
            # ATR trailing stop or opposite breakout
            if low_low > 0 and close_val > (low_low + 2.0 * atr_val):
                exit_short = True
            elif close_val > R1:  # Opposite breakout
                exit_short = True
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            long_high = high_val  # Reset extreme on new entry
            low_low = 0.0
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            low_low = low_val  # Reset extreme on new entry
            long_high = 0.0
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
            long_high = 0.0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
            low_low = 0.0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0