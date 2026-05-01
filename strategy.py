#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Williams Alligator consists of three smoothed moving averages (Jaw, Teeth, Lips)
# - Jaw (13-period SMMA, 8-bar offset): Blue line
# - Teeth (8-period SMMA, 5-bar offset): Red line  
# - Lips (5-period SMMA, 3-bar offset): Green line
# Trend: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
# Entry: Alligator aligned in trend direction + volume spike (>1.5x 20 EMA volume)
# Exit: Alligator loses alignment (lines cross)
# Works in bull/bear by following 1d EMA50 trend; volume filters false signals
# Target: 15-25 trades/year per symbol with 0.25 sizing

name = "6h_WilliamsAlligator_1dEMA50_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: Smoothed Moving Average (SMMA)
    def smma(values, period):
        """Smoothed Moving Average - similar to RMA/Wilders"""
        if len(values) < period:
            return np.full(len(values), np.nan)
        result = np.full(len(values), np.nan)
        # First value is simple average
        result[period-1] = np.mean(values[:period])
        # Subsequent values: (prev * (period-1) + current) / period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    # Calculate Alligator components
    jaw = smma(close, 13)  # 13-period SMMA
    teeth = smma(close, 8)  # 8-period SMMA
    lips = smma(close, 5)   # 5-period SMMA
    
    # Apply offsets (shift right by offset bars)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Set NaN for shifted values
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Volume confirmation: volume > 1.5 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(60, 21, 20)  # Need 1d EMA50, Alligator (21), volume EMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_aligned[i]) or np.isnan(lips[i]) or np.isnan(teeth[i]) or 
            np.isnan(jaw[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade in direction of 1d EMA50
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Alligator alignment
        bullish_aligned = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_aligned = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:  # Flat - look for new entries
            if uptrend and bullish_aligned:
                # Long: Alligator bullish + volume spike
                if volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif downtrend and bearish_aligned:
                # Short: Alligator bearish + volume spike
                if volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # No clear signal
        
        elif position == 1:  # Long position
            # Exit: Alligator loses bullish alignment
            if not bullish_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator loses bearish alignment
            if not bearish_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals