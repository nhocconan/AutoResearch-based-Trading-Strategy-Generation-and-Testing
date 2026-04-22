#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w trend filter and volume confirmation
# Uses Williams Alligator (Jaw/Teeth/Lips) on daily chart to identify trend.
# Requires alignment of Alligator lines (Lips > Teeth > Jaw for uptrend, reverse for downtrend).
# Confirmed with weekly EMA50 trend filter and volume spike.
# Designed to work in both bull and bear markets by following higher timeframe trend.
# Target: 15-25 trades/year per symbol (60-100 total) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    median_1d = (high_1d + low_1d) / 2  # Typical price
    
    # Williams Alligator: SMoothed Moving Average (SMMA) with specific periods
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    jaw_period, jaw_shift = 13, 8
    teeth_period, teeth_shift = 8, 5
    lips_period, lips_shift = 5, 3
    
    # Calculate SMMA (Smoothed Moving Average)
    def smma(series, period):
        sma = pd.Series(series).rolling(window=period, min_periods=period).mean().values
        smma_vals = np.full_like(series, np.nan, dtype=float)
        if len(series) >= period:
            smma_vals[period-1] = sma[period-1]
            for i in range(period, len(series)):
                if not np.isnan(smma_vals[i-1]):
                    smma_vals[i] = (smma_vals[i-1] * (period-1) + series[i]) / period
                else:
                    smma_vals[i] = sma[i]
        return smma_vals
    
    jaw = smma(median_1d, jaw_period)
    teeth = smma(median_1d, teeth_period)
    lips = smma(median_1d, lips_period)
    
    # Apply shifts (positive shift = move data to the right/future)
    jaw_shifted = np.roll(jaw, jaw_shift)
    teeth_shifted = np.roll(teeth, teeth_shift)
    lips_shifted = np.roll(lips, lips_shift)
    
    # Invalidate shifted values that look into the future
    jaw_shifted[:jaw_shift] = np.nan
    teeth_shifted[:teeth_shift] = np.nan
    lips_shifted[:lips_shift] = np.nan
    
    # Load 1-week data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 50-period EMA on 1w close for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike filter (20-period on daily)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align indicators to daily timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + volume spike + price > weekly EMA50
            if (lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i] and
                vol_spike[i] and close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + volume spike + price < weekly EMA50
            elif (lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i] and
                  vol_spike[i] and close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator lines cross (Lips crosses Teeth)
            if position == 1:
                if lips_aligned[i] < teeth_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if lips_aligned[i] > teeth_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Williams_Alligator_1wEMA50_Volume_Session"
timeframe = "1d"
leverage = 1.0