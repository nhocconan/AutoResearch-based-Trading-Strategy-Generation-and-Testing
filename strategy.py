#!/usr/bin/env python3
# 6h_engulfing_1d_trend_volume_v1
# Hypothesis: On 6h timeframe, enter long when bullish engulfing candle forms in direction of daily EMA trend,
# with volume confirmation (>1.5x average). Exit when price closes below EMA or opposite engulfing forms.
# Works in bull/bear markets by following higher timeframe trend. Engulfing captures strong momentum shifts.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "6h_engulfing_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily EMA trend filter (50-period)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 1:
        return np.zeros(n)
    
    ema_daily = pd.Series(df_daily['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Volume filter: volume > 1.5x 24-period average (4 days)
    vol_period = 24
    vol_ma = np.full(n, np.nan)
    vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Start from sufficient lookback
    start_idx = max(50, vol_period) + 5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_daily_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        # Bullish engulfing: current green candle fully engulfs previous red candle
        bullish_engulf = (close[i] > open_[i]) and (open_[i-1] > close[i-1]) and \
                         (close[i] >= open_[i-1]) and (open_[i] <= close[i-1])
        # Bearish engulfing: current red candle fully engulfs previous green candle
        bearish_engulf = (close[i] < open_[i]) and (open_[i-1] < close[i-1]) and \
                         (close[i] <= open_[i-1]) and (open_[i] >= close[i-1])
        
        if position == 1:  # Long position
            # Exit: price closes below EMA or bearish engulfing forms
            if close[i] < ema_daily_aligned[i] or bearish_engulf:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above EMA or bullish engulfing forms
            if close[i] > ema_daily_aligned[i] or bullish_engulf:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade with volume confirmation
            if volume_filter:
                # Enter long: bullish engulfing with uptrend
                if bullish_engulf and close[i] > ema_daily_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short: bearish engulfing with downtrend
                elif bearish_engulf and close[i] < ema_daily_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals