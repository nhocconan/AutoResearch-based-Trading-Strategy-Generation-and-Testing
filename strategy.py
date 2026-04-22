#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# Uses Williams Alligator (Jaw/Teeth/Lips) on 12h for trend direction
# Long when Lips > Teeth > Jaw with 1d uptrend and volume spike
# Short when Lips < Teeth < Jaw with 1d downtrend and volume spike
# Designed for 12h timeframe to target 12-37 trades/year per symbol.
# Williams Alligator catches trends early while avoiding whipsaws in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Williams Alligator calculation (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams Alligator on 12h data
    # Jaw (Blue): 13-period SMMA smoothed 8 bars ahead
    # Teeth (Red): 8-period SMMA smoothed 5 bars ahead
    # Lips (Green): 5-period SMMA smoothed 3 bars ahead
    def smma(data, period):
        """Smoothed Moving Average"""
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CLOSE) / PERIOD
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw_12h = smma(smma(close_12h, 13), 8)  # SMMA(13) then smoothed 8
    teeth_12h = smma(smma(close_12h, 8), 5)  # SMMA(8) then smoothed 5
    lips_12h = smma(smma(close_12h, 5), 3)   # SMMA(5) then smoothed 3
    
    # Align Williams Alligator lines to 12h timeframe
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # 1d EMA(34) for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike filter (20-period on 12h data)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Higher threshold for fewer, higher quality trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]) or 
            np.isnan(lips_12h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + 1d uptrend + volume spike
            if (lips_12h_aligned[i] > teeth_12h_aligned[i] > jaw_12h_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + 1d downtrend + volume spike
            elif (lips_12h_aligned[i] < teeth_12h_aligned[i] < jaw_12h_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator lines intertwine (no clear trend) or trend reversal
            if position == 1:
                # Exit on bearish alignment or trend reversal
                if (lips_12h_aligned[i] < teeth_12h_aligned[i] or 
                    teeth_12h_aligned[i] < jaw_12h_aligned[i] or
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on bullish alignment or trend reversal
                if (lips_12h_aligned[i] > teeth_12h_aligned[i] or 
                    teeth_12h_aligned[i] > jaw_12h_aligned[i] or
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA34_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0