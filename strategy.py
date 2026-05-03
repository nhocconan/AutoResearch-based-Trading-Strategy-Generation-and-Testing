#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA50 trend filter + volume confirmation.
# Long when Alligator jaws < teeth < lips (bullish alignment) AND 1d close > 1d EMA50 AND 12h volume > 1.5x 20-period volume MA.
# Short when Alligator jaws > teeth > lips (bearish alignment) AND 1d close < 1d EMA50 AND 12h volume > 1.5x 20-period volume MA.
# Exit when Alligator alignment reverses or trend changes.
# Uses session filter (08-20 UTC) to avoid low-liquidity periods. Position size 0.25.
# Designed for 12h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with strict entry conditions.
# Williams Alligator identifies trend phases via smoothed medians, 1d EMA50 filters for higher-timeframe alignment, volume confirms participation.
# Works in both bull and bear markets by only trading in the direction of the 1d trend when volume confirms and Alligator is aligned.

name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike_Session"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 12h timeframe
    # Alligator: Jaw (13-period smoothed median of typical price, 8-bar shift)
    #          Teeth (8-period smoothed median of typical price, 5-bar shift)
    #          Lips (5-period smoothed median of typical price, 3-bar shift)
    typical_price = (high + low + close) / 3.0
    
    # Calculate medians using rolling window
    def rolling_median(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).median().values
    
    median_tp = rolling_median(typical_price, 13)  # Jaw base
    jaw = np.roll(median_tp, 8)  # 8-bar shift
    jaw[:8] = np.nan
    
    median_tp_teeth = rolling_median(typical_price, 8)  # Teeth base
    teeth = np.roll(median_tp_teeth, 5)  # 5-bar shift
    teeth[:5] = np.nan
    
    median_tp_lips = rolling_median(typical_price, 5)  # Lips base
    lips = np.roll(median_tp_lips, 3)  # 3-bar shift
    lips[:3] = np.nan
    
    # Calculate 12h volume 20-period MA for spike detection
    volume_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(volume_ma_12h[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        
        # Volume spike condition: current 12h volume > 1.5x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_12h[i] * 1.5)
        
        # Alligator alignment conditions
        bullish_alignment = jaw[i] < teeth[i] and teeth[i] < lips[i]   # Jaws < Teeth < Lips
        bearish_alignment = jaw[i] > teeth[i] and teeth[i] > lips[i]   # Jaws > Teeth > Lips
        
        # 1d trend conditions
        trend_up = close_val > ema_50_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_50_1d_aligned[i]  # 1d downtrend
        
        if position == 0:
            # Long: Bullish Alligator alignment AND 1d uptrend AND volume spike AND session
            if bullish_alignment and trend_up and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment AND 1d downtrend AND volume spike AND session
            elif bearish_alignment and trend_down and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment turns bearish OR trend changes
            if not bullish_alignment or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment turns bullish OR trend changes
            if not bearish_alignment or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals