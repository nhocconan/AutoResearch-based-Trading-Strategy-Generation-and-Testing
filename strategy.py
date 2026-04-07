#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h 4h EMA Trend with Volume Confirmation v1
# Hypothesis: In trending markets, price respects the 4h EMA(21) as dynamic support/resistance.
# During pullbacks to the EMA with volume confirmation, enter in trend direction.
# Works in both bull/bear as it follows the 4h trend. Volume filters out false breakouts.
# Target: 15-37 trades/year (60-150 over 4 years).

name = "1h_4h_ema_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # Calculate EMA(21) on 4h close
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False).mean().values
    
    # Align EMA to 1h timeframe (shifted by 1 for completed bars)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC (active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(40, n):
        # Skip if required data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below EMA (trend change)
            if close[i] < ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above EMA (trend change)
            if close[i] > ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Determine trend direction from EMA slope (using 3-period change)
                if i >= 3:
                    ema_slope = ema_4h_aligned[i] - ema_4h_aligned[i-3]
                    
                    # Long: uptrend (EMA rising) + pullback to EMA
                    if ema_slope > 0 and low[i] <= ema_4h_aligned[i] and close[i] > ema_4h_aligned[i]:
                        # Additional: close in upper half of hourly range
                        hourly_range = high[i] - low[i]
                        if hourly_range > 0:
                            close_position = (close[i] - low[i]) / hourly_range
                            if close_position > 0.5:
                                position = 1
                                signals[i] = 0.20
                    
                    # Short: downtrend (EMA falling) + pullback to EMA
                    elif ema_slope < 0 and high[i] >= ema_4h_aligned[i] and close[i] < ema_4h_aligned[i]:
                        # Additional: close in lower half of hourly range
                        hourly_range = high[i] - low[i]
                        if hourly_range > 0:
                            close_position = (close[i] - low[i]) / hourly_range
                            if close_position < 0.5:
                                position = -1
                                signals[i] = -0.20
    
    return signals