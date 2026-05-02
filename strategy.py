#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d EMA34 trend filter and volume confirmation
# Uses 1d EMA34 for trend direction (price > EMA34 = bull, price < EMA34 = bear)
# Williams Alligator (jaw=13, teeth=8, lips=5) from 6h data for entry signals
# Long when lips cross above jaw in bull trend, short when lips cross below jaw in bear trend
# Volume confirmation (1.5x 20-period average) ensures institutional participation
# Designed for low trade frequency to minimize fee drag (critical for 6h timeframe)
# Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe
# Works in bull markets via trend-aligned Alligator signals, in bear via inverse signals
# Williams Alligator is effective in trending markets and avoids whipsaws in ranging conditions

name = "6h_WilliamsAlligator_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) - index is DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator from 6h data
    # Jaw: 13-period SMMA (smoothed moving average) of median price
    # Teeth: 8-period SMMA of median price
    # Lips: 5-period SMMA of median price
    median_price = (high + low) / 2.0
    
    # SMMA calculation (similar to Wilder's smoothing)
    def smma(source, period):
        result = np.full_like(source, np.nan, dtype=np.float64)
        if len(source) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 1d EMA34
        bull_trend = close[i] > ema_34_1d_aligned[i]
        bear_trend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Lips cross above Jaw in bull trend + volume confirm
            if lips[i] > jaw[i] and lips[i-1] <= jaw[i-1] and bull_trend and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lips cross below Jaw in bear trend + volume confirm
            elif lips[i] < jaw[i] and lips[i-1] >= jaw[i-1] and bear_trend and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Lips cross below Teeth or trend reversal
            if lips[i] < teeth[i] or (close[i] < ema_34_1d_aligned[i] and bear_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Lips cross above Teeth or trend reversal
            if lips[i] > teeth[i] or (close[i] > ema_34_1d_aligned[i] and bull_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals