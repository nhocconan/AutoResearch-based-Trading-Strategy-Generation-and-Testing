#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d ADX trend filter + volume confirmation.
# Williams Alligator (JAWS=13, TEETH=8, LIPS=5 SMMA) identifies trend direction.
# ADX > 25 confirms trending market (avoid ranging).
# Volume > 1.5x 20-period MA confirms momentum.
# Long when GATOR > TEETH > LIPS (bullish alignment) + ADX > 25 + volume spike.
# Short when LIPS > TEETH > JAWS (bearish alignment) + ADX > 25 + volume spike.
# Designed for 12h timeframe to capture multi-day trends with low frequency.
# Target: 12-37 trades/year per symbol (48-148 total) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Williams Alligator (SMMA)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Smoothed Moving Average (SMMA) calculation
    def smma(source, length):
        sma = np.full_like(source, np.nan, dtype=float)
        sma[length-1] = np.mean(source[:length])
        for i in range(length, len(source)):
            if not np.isnan(sma[i-1]):
                sma[i] = (sma[i-1] * (length-1) + source[i]) / length
        return sma
    
    jaws = smma(close_12h, 13)  # Blue line
    teeth = smma(close_12h, 8)   # Red line
    lips = smma(close_12h, 5)    # Green line
    
    # Load 1d data for ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range (TR)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1d[0] - low_1d[0]  # First TR
    
    # Directional Movement
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed TR, +DM, -DM (14-period)
    def smooth_rma(source, length):
        rma = np.full_like(source, np.nan, dtype=float)
        rma[length-1] = np.nansum(source[:length]) / length
        for i in range(length, len(source)):
            if not np.isnan(rma[i-1]):
                rma[i] = (rma[i-1] * (length-1) + source[i]) / length
        return rma
    
    atr = smooth_rma(tr, 14)
    plus_di = 100 * smooth_rma(plus_dm, 14) / atr
    minus_di = 100 * smooth_rma(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth_rma(dx, 14)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20
    
    # Align indicators to 12h timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_12h, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish alignment: JAWS < TEETH < LIPS
            bullish = jaws_aligned[i] < teeth_aligned[i] < lips_aligned[i]
            # Bearish alignment: LIPS < TEETH < JAWS
            bearish = lips_aligned[i] < teeth_aligned[i] < jaws_aligned[i]
            
            if bullish and adx_aligned[i] > 25 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            elif bearish and adx_aligned[i] > 25 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: alignment breaks or ADX < 20 (trend weakening)
            if position == 1:
                bullish = jaws_aligned[i] < teeth_aligned[i] < lips_aligned[i]
                if not bullish or adx_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                bearish = lips_aligned[i] < teeth_aligned[i] < jaws_aligned[i]
                if not bearish or adx_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_ADX_Trend_Volume_Spike"
timeframe = "12h"
leverage = 1.0