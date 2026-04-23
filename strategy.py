#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 1d EMA34 trend filter and volume confirmation.
Long when Alligator jaws < teeth < lips (bullish alignment) AND close > 1d EMA34 AND volume > 1.8x 20-period average.
Short when Alligator jaws > teeth > lips (bearish alignment) AND close < 1d EMA34 AND volume > 1.8x 20-period average.
Exit when Alligator alignment reverses OR ATR trailing stop (2.0*ATR from extreme).
Uses discrete position sizing (0.25) to target 20-40 trades/year on 4h timeframe.
Williams Alligator (SMAs with specific periods) provides trend identification proven in choppy BTC/ETH markets.
4h timeframe balances noise reduction and trade frequency for both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator (13,8,5 SMAs with specific shifts)
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    def smma(source, period):
        """Smoothed Moving Average"""
        if len(source) < period:
            return np.full_like(source, np.nan, dtype=float)
        result = np.full_like(source, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Apply shifts: Jaw shifted 8, Teeth shifted 5, Lips shifted 3
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted values that would look ahead
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # Need enough for Alligator calculation + volume + ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema34_val = ema34_1d_aligned[i]
        jaw_val = jaw_shifted[i]
        teeth_val = teeth_shifted[i]
        lips_val = lips_shifted[i]
        
        if position == 0:
            # Bullish Alligator alignment: Jaw < Teeth < Lips
            bullish_alignment = jaw_val < teeth_val < lips_val
            # Bearish Alligator alignment: Jaw > Teeth > Lips
            bearish_alignment = jaw_val > teeth_val > lips_val
            
            # Long: Bullish alignment AND uptrend (price > EMA34) AND volume spike (1.8x avg)
            if bullish_alignment and close[i] > ema34_val and volume[i] > 1.8 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Bearish alignment AND downtrend (price < EMA34) AND volume spike (1.8x avg)
            elif bearish_alignment and close[i] < ema34_val and volume[i] > 1.8 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Alligator alignment reverses
            bullish_alignment = jaw_val < teeth_val < lips_val
            bearish_alignment = jaw_val > teeth_val > lips_val
            
            if position == 1 and not bullish_alignment:
                exit_signal = True
            elif position == -1 and not bearish_alignment:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsAlligator_1dEMA34_Trend_VolumeConfirmation_AlignmentExit_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0