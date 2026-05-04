#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) with 1w EMA50 trend filter and volume confirmation (>1.8x 12h EMA volume)
# Williams Alligator identifies trend presence/absence via SMAs (13,8,5) with future shifts - effective in both bull/bear
# 1w EMA50 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws in ranging markets
# Volume confirmation filters false signals (>1.8x average volume) - balances sensitivity and trade frequency
# Discrete sizing 0.28 minimizes fee churn while maintaining profitability
# Target: 50-150 total trades over 4 years = 12-37/year for 12h timeframe
# Works in bull markets (Alligator eating - Lips above Teeth above Jaw) and bear markets (Alligator sleeping - converging lines)
# Focus on BTC/ETH by requiring 1w trend alignment (avoids SOL-only bias)

name = "12h_WilliamsAlligator_1wEMA50_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough data for EMA50 calculation
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50) trend filter from prior completed 1w bar
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_shifted = np.roll(ema_50_1w, 1)
    ema_50_1w_shifted[0] = np.nan
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w_shifted)
    
    # Volume confirmation: 20-period EMA of volume (12h timeframe)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Williams Alligator components from prior completed 12h bar
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    close_series = pd.Series(close)
    
    # SMMA calculation (Smoothed Moving Average)
    def smma(series, period):
        sma = series.rolling(window=period, min_periods=period).mean()
        result = np.full(len(series), np.nan)
        if len(series) >= period:
            result[period-1] = sma.iloc[period-1]
            for i in range(period, len(series)):
                result[i] = (result[i-1] * (period-1) + series.iloc[i]) / period
        return result
    
    jaw = smma(close_series, 13)
    teeth = smma(close_series, 8)
    lips = smma(close_series, 5)
    
    # Apply Alligator shifts (Jaw: +8, Teeth: +5, Lips: +3)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Shift by 1 to use only prior completed 12h bar (no look-ahead)
    jaw_shifted = np.roll(jaw_shifted, 1)
    teeth_shifted = np.roll(teeth_shifted, 1)
    lips_shifted = np.roll(lips_shifted, 1)
    jaw_shifted[0] = np.nan
    teeth_shifted[0] = np.nan
    lips_shifted[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Lips > Teeth > Jaw (Alligator eating up) AND price > 1w EMA50 AND volume spike
            if lips_shifted[i] > teeth_shifted[i] and teeth_shifted[i] > jaw_shifted[i] and close[i] > ema_50_1w_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = 0.28
                position = 1
            # Short conditions: Lips < Teeth < Jaw (Alligator eating down) AND price < 1w EMA50 AND volume spike
            elif lips_shifted[i] < teeth_shifted[i] and teeth_shifted[i] < jaw_shifted[i] and close[i] < ema_50_1w_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = -0.28
                position = -1
        elif position == 1:
            # Exit long: Alligator sleeping (convergence) OR price crosses below 1w EMA50
            if (abs(lips_shifted[i] - teeth_shifted[i]) < 0.001 * close[i] and abs(teeth_shifted[i] - jaw_shifted[i]) < 0.001 * close[i]) or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        elif position == -1:
            # Exit short: Alligator sleeping (convergence) OR price crosses above 1w EMA50
            if (abs(lips_shifted[i] - teeth_shifted[i]) < 0.001 * close[i] and abs(teeth_shifted[i] - jaw_shifted[i]) < 0.001 * close[i]) or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals