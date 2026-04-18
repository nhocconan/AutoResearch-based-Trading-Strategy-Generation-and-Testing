#!/usr/bin/env python3
"""
4h_Gap_Fade_Retest
Fade gaps with mean reversion on 4h chart.
- Gap defined: open > previous close + 0.5% (gap up) or open < previous close - 0.5% (gap down)
- Fade condition: price retraces 50% of gap toward previous close
- Volume confirmation: current volume > 1.5x 20-bar average
- Trend filter: price above/below 50 EMA to avoid counter-trend fades
- Exit: price reaches previous close or opposite signal
- Designed for 20-40 trades/year per symbol
Works in bull (fade gap ups in uptrend) and bear (fade gap downs in downtrend) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_ = prices['open'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 50 EMA on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Previous close for gap calculation
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    
    # Gap size and direction
    gap_up = (open_ - prev_close) > (prev_close * 0.005)  # gap up > 0.5%
    gap_down = (open_ - prev_close) < (-prev_close * 0.005)  # gap down < -0.5%
    
    # Gap midpoint (50% retracement level)
    gap_mid = prev_close + (open_ - prev_close) * 0.5
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need 50 for EMA + buffer
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(gap_mid[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 1d EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Fade condition: price retraced to gap midpoint
        if gap_up[i]:
            faded_to_mid = low[i] <= gap_mid[i]  # gap up faded if low touches midpoint
        elif gap_down[i]:
            faded_to_mid = high[i] >= gap_mid[i]  # gap down faded if high touches midpoint
        else:
            faded_to_mid = False
        
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long fade: gap down faded + above EMA + volume
            if gap_down[i] and faded_to_mid and price_above_ema and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short fade: gap up faded + below EMA + volume
            elif gap_up[i] and faded_to_mid and price_below_ema and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches previous close or gap fills completely
            if high[i] >= prev_close[i] or gap_up[i]:  # gap filled or new gap up
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches previous close or gap fills completely
            if low[i] <= prev_close[i] or gap_down[i]:  # gap filled or new gap down
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Gap_Fade_Retest"
timeframe = "4h"
leverage = 1.0