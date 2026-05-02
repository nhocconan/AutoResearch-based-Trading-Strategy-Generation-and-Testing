#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume confirmation
# Uses 12h timeframe for signal generation with Williams Alligator (Jaw/Teeth/Lips)
# 1d EMA34 provides multi-timeframe trend filter to avoid counter-trend trades
# Volume confirmation (1.5x 20-period average) ensures institutional participation
# Chop regime filter from 12h timeframe avoids ranging markets (CHOP > 61.8 = range)
# Discrete position sizing (0.25) minimizes fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 12h timeframe
# Williams Alligator catches trends early; EMA34 filter ensures trend alignment
# Volume confirmation reduces false breakouts; chop filter avoids whipsaws in ranging markets
# Designed for low trade frequency to minimize fee drag (critical for 12h timeframe)

name = "12h_WilliamsAlligator_1dEMA34_VolumeS_ChopFilter_v1"
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
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaw (Blue): 13-period SMMA, shifted 8 bars ahead
    # Teeth (Red): 8-period SMMA, shifted 5 bars ahead  
    # Lips (Green): 5-period SMMA, shifted 3 bars ahead
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CLOSE) / PERIOD
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift as per Alligator specification
    jaw = np.roll(jaw, 8)  # shifted 8 bars ahead
    teeth = np.roll(teeth, 5)  # shifted 5 bars ahead
    lips = np.roll(lips, 3)  # shifted 3 bars ahead
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    # Calculate 12h Chopiness Index (14) - trending when < 38.2, ranging when > 61.8
    # True Range
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR14
    atr1 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Max high and min low over 14 periods
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(ATR14)/ (max(high)-min(low)) over 14 periods) / log10(14)
    # Avoid division by zero and handle NaN
    denominator = max_high - min_low
    chop = np.full_like(close, np.nan)
    valid_mask = (denominator > 0) & (~np.isnan(atr1)) & (~np.isnan(denominator))
    chop[valid_mask] = 100 * np.log10(atr1[valid_mask] * 14 / denominator[valid_mask]) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_confirm[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when Chop < 61.8 (not strongly ranging)
        if chop[i] > 61.8:
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Alligator is bullish when Lips > Teeth > Jaw (green above red above blue)
            # Alligator is bearish when Jaw > Teeth > Lips (blue above red above green)
            if lips[i] > teeth[i] and teeth[i] > jaw[i]:  # Bullish alignment
                if close[i] > ema_34_1d_aligned[i] and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
            elif jaw[i] > teeth[i] and teeth[i] > lips[i]:  # Bearish alignment
                if close[i] < ema_34_1d_aligned[i] and volume_confirm[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator turns bearish (Jaw > Teeth > Lips) or reverse signal
            if jaw[i] > teeth[i] and teeth[i] > lips[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator turns bullish (Lips > Teeth > Jaw) or reverse signal
            if lips[i] > teeth[i] and teeth[i] > jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals