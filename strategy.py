#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + 1d Williams Alligator combo.
# ADX(14) > 25 signals trend presence (avoids chop). 
# Williams Alligator: Jaw(13), Teeth(8), Lips(5) SMAs on median price.
# Long when price > Teeth and Teeth > Lips and ADX rising.
# Short when price < Teeth and Teeth < Lips and ADX rising.
# Uses 1d Alligator for higher timeframe trend filter, ADX on 6h for entry timing.
# Target: 25-40 trades/year by requiring strong trend + alignment.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d
    median_price_1d = (df_1d['high'].values + df_1d['low'].values) / 2
    jaw = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values  # Blue line (13)
    teeth = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values    # Red line (8)
    lips = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values     # Green line (5)
    
    # Align Alligator lines to 6h
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate ADX on 6h
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+, DM- (Wilder smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Adx rising: current > previous
    adx_rising = adx > np.roll(adx, 1)
    adx_rising[0] = False
    
    # Alligator alignment signals
    # Bullish: price > teeth and teeth > lips
    bullish_align = (close > teeth_1d_aligned) & (teeth_1d_aligned > lips_1d_aligned)
    # Bearish: price < teeth and teeth < lips
    bearish_align = (close < teeth_1d_aligned) & (teeth_1d_aligned < lips_1d_aligned)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after ADX warmup
        # Skip if data not ready
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(adx[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bullish alignment + ADX > 25 + ADX rising
            if bullish_align[i] and adx[i] > 25 and adx_rising[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish alignment + ADX > 25 + ADX rising
            elif bearish_align[i] and adx[i] > 25 and adx_rising[i]:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: loss of bullish alignment OR ADX < 20
                if not bullish_align[i] or adx[i] < 20:
                    exit_signal = True
            elif position == -1:
                # Exit short: loss of bearish alignment OR ADX < 20
                if not bearish_align[i] or adx[i] < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ADX_WilliamsAlligator"
timeframe = "6h"
leverage = 1.0