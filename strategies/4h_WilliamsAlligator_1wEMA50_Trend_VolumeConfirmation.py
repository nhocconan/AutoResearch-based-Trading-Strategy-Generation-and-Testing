#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator trend system with 1w trend filter and volume confirmation.
# Uses Williams Alligator (Jaw/Teeth/Lips) for trend direction and entry signals.
# Long when Lips > Teeth > Jaw (bullish alignment) with price above Teeth and volume confirmation.
# Short when Lips < Teeth < Jaw (bearish alignment) with price below Teeth and volume confirmation.
# Uses weekly EMA(50) as higher timeframe trend filter to avoid counter-trend trades.
# Designed for 4h timeframe to target 20-50 trades/year per symbol.
# Works in bull/bear via multi-timeframe trend alignment and volatility-based signals.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for higher timeframe trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate Williams Alligator on 4h data
    # Jaw (Blue): 13-period SMMA, shifted 8 bars forward
    # Teeth (Red): 8-period SMMA, shifted 5 bars forward  
    # Lips (Green): 5-period SMMA, shifted 3 bars forward
    # SMMA = Smoothed Moving Average (similar to Wilder's smoothing)
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # Jaw: 13-period SMMA
    teeth = smma(close, 8)  # Teeth: 8-period SMMA
    lips = smma(close, 5)   # Lips: 5-period SMMA
    
    # Shift as per Alligator definition
    jaw_shifted = np.full_like(jaw, np.nan)
    teeth_shifted = np.full_like(teeth, np.nan)
    lips_shifted = np.full_like(lips, np.nan)
    
    if len(jaw) > 8:
        jaw_shifted[8:] = jaw[:-8]
    if len(teeth) > 5:
        teeth_shifted[5:] = teeth[:-5]
    if len(lips) > 3:
        lips_shifted[3:] = lips[:-3]
    
    # 1w EMA(50) for higher timeframe trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.8 * vol_ma20  # Moderate threshold for balance
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(lips_shifted[i]) or np.isnan(teeth_shifted[i]) or 
            np.isnan(jaw_shifted[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish alignment: Lips > Teeth > Jaw
            bullish_alignment = (lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i])
            # Bearish alignment: Lips < Teeth < Jaw
            bearish_alignment = (lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i])
            
            # Long: bullish alignment + price above Teeth + weekly uptrend + volume spike
            if (bullish_alignment and 
                close[i] > teeth_shifted[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment + price below Teeth + weekly downtrend + volume spike
            elif (bearish_alignment and 
                  close[i] < teeth_shifted[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit on bearish alignment or price below Jaw or weekly trend reversal
                bearish_exit = (lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i])
                if (bearish_exit or close[i] < jaw_shifted[i] or close[i] < ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on bullish alignment or price above Jaw or weekly trend reversal
                bullish_exit = (lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i])
                if (bullish_exit or close[i] > jaw_shifted[i] or close[i] > ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1wEMA50_Trend_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0