#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R mean reversion with 1w trend filter and volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for trend direction (price above/below 50-week EMA) and 1d for volume spike.
- Williams %R(14): Oversold < -80 for long, overbought > -20 for short.
- Entry: Long when %R < -80 AND price > 50-week EMA AND volume > 2.0 * 20-period average volume.
         Short when %R > -20 AND price < 50-week EMA AND volume > 2.0 * 20-period average volume.
- Exit: Opposite %R level (%R > -50 for long exit, %R < -50 for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in bull markets by buying dips in uptrend, and in bear markets by selling rallies in downtrend.
- Volume spike confirms conviction; 1w EMA filter avoids trading against the major trend.
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
    
    # Calculate 1w 50-period EMA for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 1d volume average for confirmation (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for volume MA
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 6h Williams %R(14)
    williams_window = 14
    highest_high = pd.Series(high).rolling(window=williams_window, min_periods=williams_window).max().values
    lowest_low = pd.Series(low).rolling(window=williams_window, min_periods=williams_window).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close) / (highest_high - lowest_low)) * -100,
        -50  # Default to neutral when range is zero
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(williams_window, 50)  # Need 14 for Williams, 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(williams_r[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_williams = williams_r[i]
        
        # Trend filter: price above/below 50-week EMA
        uptrend = curr_close > ema50_1w_aligned[i]
        downtrend = curr_close < ema50_1w_aligned[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Exit conditions: opposite Williams %R level
        if position != 0:
            # Exit long: Williams %R > -50 (recovering from oversold)
            if position == 1:
                if curr_williams > -50:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Williams %R < -50 (recovering from overbought)
            elif position == -1:
                if curr_williams < -50:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R extremes with trend and volume filters
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND uptrend AND volume confirmation
            long_condition = (curr_williams < -80 and 
                            uptrend and
                            volume_confirm)
            
            # Short: Williams %R > -20 (overbought) AND downtrend AND volume confirmation
            short_condition = (curr_williams > -20 and 
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

name = "6h_WilliamsR_1wEMA50Trend_1dVolumeSpike_v1"
timeframe = "6h"
leverage = 1.0