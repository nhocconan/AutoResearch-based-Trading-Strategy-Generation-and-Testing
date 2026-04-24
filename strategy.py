#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R mean reversion with 1d EMA trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA trend filter (avoid counter-trend trades).
- Williams %R(14): Oversold < -80 for long, overbought > -20 for short.
- Trend filter: EMA(34) on 1d - price above EMA = uptrend (long bias), below = downtrend (short bias).
- Volume confirmation: current volume > 1.5 * 20-period average volume.
- Exit: Opposite Williams %R level (long exit when %R > -50, short exit when %R < -50).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in bull markets by taking longs in uptrend, in bear markets by taking shorts in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA(34) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Williams %R(14) on 12h timeframe
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close) / (highest_high - lowest_low)) * -100,
        -50  # neutral when range is zero
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 34)  # Need 14 for Williams %R, 34 for EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(williams_r[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_wr = williams_r[i]
        
        # Trend filter: price above EMA34 = uptrend (long bias), below = downtrend (short bias)
        uptrend = curr_close > ema34_aligned[i]
        downtrend = curr_close < ema34_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Exit conditions: opposite Williams %R level
        if position != 0:
            # Exit long: Williams %R > -50 (exiting oversold territory)
            if position == 1:
                if curr_wr > -50:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Williams %R < -50 (exiting overbought territory)
            elif position == -1:
                if curr_wr < -50:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R extreme with trend and volume filters
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND uptrend AND volume confirmation
            long_condition = (curr_wr < -80 and 
                            uptrend and
                            volume_confirm)
            
            # Short: Williams %R > -20 (overbought) AND downtrend AND volume confirmation
            short_condition = (curr_wr > -20 and 
                             downtrend and
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

name = "12h_WilliamsR_MeanReversion_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0