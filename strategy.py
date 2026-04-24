#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R with 1d EMA34 trend filter and volume spike.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA34 trend direction.
- Williams %R: momentum oscillator (-100 to 0), long when < -80 (oversold), short when > -20 (overbought).
- Entry: Long when Williams %R < -80 AND price > 1d EMA34 AND volume > 2.0 * 20-period average volume.
         Short when Williams %R > -20 AND price < 1d EMA34 AND volume > 2.0 * 20-period average volume.
- Exit: Opposite Williams %R signal (Williams %R > -50 for long exit, < -50 for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Williams %R identifies exhaustion points; volume spike confirms conviction; EMA34 filter ensures trend alignment.
- Works in ranging markets (mean reversion at extremes) and avoids counter-trend trades in strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def williams_r(high, low, close, period):
    """Williams %R oscillator: (Highest High - Close) / (Highest High - Lowest Low) * -100"""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return wr.values

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema34_1d = ema(df_1d['close'].values, 34)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 4h Williams %R (14-period)
    if len(prices) < 14:
        return np.zeros(n)
    
    wr_14 = williams_r(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)  # Need 34 for EMA, 20 for volume MA, 14 for Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(wr_14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_wr = wr_14[i]
        
        # Exit conditions: opposite Williams %R signal
        if position != 0:
            # Exit long: Williams %R > -50 (momentum weakening)
            if position == 1:
                if curr_wr > -50:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Williams %R < -50 (momentum weakening)
            elif position == -1:
                if curr_wr < -50:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R with trend filter and volume confirmation
        if position == 0:
            # Volume confirmation: current volume > 2.0 * 20-period average volume
            volume_confirm = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
            
            # Long: Williams %R < -80 (oversold) AND price > 1d EMA34
            long_condition = (curr_wr < -80 and 
                            curr_close > ema34_1d_aligned[i] and
                            volume_confirm)
            
            # Short: Williams %R > -20 (overbought) AND price < 1d EMA34
            short_condition = (curr_wr > -20 and 
                             curr_close < ema34_1d_aligned[i] and
                             volume_confirm)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0