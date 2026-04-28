#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + 1w trend filter + volume spike confirmation.
# Enter long when price > Alligator Jaw (13-period SMMA) and 1w EMA34 trending up and volume > 2.0x 20-bar average.
# Enter short when price < Alligator Jaw and 1w EMA34 trending down and volume > 2.0x 20-bar average.
# Exit when price crosses the Alligator Jaw or 1w EMA34.
# Uses discrete position sizing (0.30) to balance return and fee drag.
# Target: 30-100 total trades over 4 years (7-25/year) to avoid excessive fee drag.
# Williams Alligator (SMMA-based) provides smooth trend detection; 1w EMA34 filters for weekly trend alignment;
# Volume spike confirms institutional participation in breakouts. Works in both bull and bear markets by
# following the dominant weekly trend while using the Alligator for precise 1d entry timing.

name = "1d_Williams_Alligator_1wEMA34_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    close_1w = df_1w['close'].values
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Calculate Williams Alligator (SMMA-based) on 1d timeframe
    # Jaw: 13-period SMMA of median price, smoothed 8 bars
    # Teeth: 8-period SMMA of median price, smoothed 5 bars
    # Lips: 5-period SMMA of median price, smoothed 3 bars
    # We use Jaw (13,8) as the main trend indicator
    median_price = (high + low) / 2
    
    # SMMA (Smoothed Moving Average) calculation
    def smma(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan)
        result = np.full_like(data, np.nan)
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Jaw: 13-period SMMA smoothed 8 bars
    jaw_raw = smma(median_price, 13)
    jaw = smma(jaw_raw, 8) if not np.all(np.isnan(jaw_raw)) else np.full_like(jaw_raw, np.nan)
    
    # Teeth: 8-period SMMA smoothed 5 bars
    teeth_raw = smma(median_price, 8)
    teeth = smma(teeth_raw, 5) if not np.all(np.isnan(teeth_raw)) else np.full_like(teeth_raw, np.nan)
    
    # Lips: 5-period SMMA smoothed 3 bars
    lips_raw = smma(median_price, 5)
    lips = smma(lips_raw, 3) if not np.all(np.isnan(lips_raw)) else np.full_like(lips_raw, np.nan)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1w EMA34 trend: slope over 3 periods
        if i >= 3:
            ema_slope = (ema_34_aligned[i] - ema_34_aligned[i-3]) / 3
            ema_trend_up = ema_slope > 0
            ema_trend_down = ema_slope < 0
        else:
            ema_trend_up = False
            ema_trend_down = False
        
        # Alligator conditions: price relative to Jaw
        price_above_jaw = close[i] > jaw[i]
        price_below_jaw = close[i] < jaw[i]
        
        # Exit conditions: price crosses Jaw or 1w EMA34
        exit_long = close[i] < jaw[i] or close[i] < ema_34_aligned[i]
        exit_short = close[i] > jaw[i] or close[i] > ema_34_aligned[i]
        
        # Handle entries and exits
        if price_above_jaw and ema_trend_up and vol_confirm and position <= 0:
            signals[i] = 0.30
            position = 1
        elif price_below_jaw and ema_trend_down and vol_confirm and position >= 0:
            signals[i] = -0.30
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals