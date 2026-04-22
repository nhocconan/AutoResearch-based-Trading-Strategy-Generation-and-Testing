#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and 1w volume confirmation
# Williams Alligator uses three smoothed moving averages (Jaw, Teeth, Lips) to identify trends.
# When lines are intertwined (no clear trend), market is sleeping; when they diverge, trend is forming.
# Long when Lips > Teeth > Jaw (bullish alignment), Short when Lips < Teeth < Jaw (bearish alignment).
# Filtered by 1d EMA50 trend and 1w volume spike (>1.5x 20-week average) to avoid false signals.
# Designed for 12h timeframe targeting 12-30 trades/year with low frequency to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Load 1w data for volume confirmation (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    volume_1w = df_1w['volume'].values
    
    # Williams Alligator: three SMMA (Smoothed Moving Average) lines
    # Jaw: SMMA(13, 8), Teeth: SMMA(8, 5), Lips: SMMA(5, 3)
    def smma(data, period, shift):
        # Smoothed Moving Average: similar to EMA but with different smoothing
        # SMMA today = (SMMA yesterday * (period-1) + price today) / period
        result = np.full_like(data, np.nan, dtype=np.float64)
        if len(data) >= period:
            # Initialize with SMA
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        # Apply shift
        shifted_result = np.full_like(data, np.nan, dtype=np.float64)
        if shift < len(data):
            shifted_result[shift:] = result[:-shift] if shift > 0 else result
        return shifted_result
    
    jaw = smma(close, 13, 8)
    teeth = smma(close, 8, 5)
    lips = smma(close, 5, 3)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1w volume 20-period average for spike detection
    vol_avg_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + 1d uptrend + 1w volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and
                close[i] > ema_50_1d_aligned[i] and
                volume[i] > 1.5 * vol_avg_20_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + 1d downtrend + 1w volume spike
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and
                  close[i] < ema_50_1d_aligned[i] and
                  volume[i] > 1.5 * vol_avg_20_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator lines re-intertwine (market sleeping) or trend reversal
            if position == 1:
                # Exit when Lips <= Teeth or Teeth <= Jaw (lines intertwining) or trend reversal
                if (lips[i] <= teeth[i] or teeth[i] <= jaw[i] or
                    close[i] < ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit when Lips >= Teeth or Teeth >= Jaw (lines intertwining) or trend reversal
                if (lips[i] >= teeth[i] or teeth[i] >= jaw[i] or
                    close[i] > ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_1wVolSpike"
timeframe = "12h"
leverage = 1.0