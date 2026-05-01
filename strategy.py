#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d Elder Ray combination for trend detection
# Long when Alligator jaws-teeth-lips are aligned bullish (jaws < teeth < lips) AND Bull Power > 0
# Short when Alligator aligned bearish (jaws > teeth > lips) AND Bear Power < 0
# Uses volume confirmation: current volume > 1.5x 6h volume median
# Discrete sizing 0.25. Williams Alligator uses SMAs of median price (hlc3) with specific periods.
# Elder Ray measures bull/bear power relative to 13-period EMA.
# Target: 12-25 trades/year on 6h timeframe (~50-100 total over 4 years).
# Works in both bull and bear by requiring both trend alignment and power confirmation.

name = "6h_WilliamsAlligator_1dElderRay_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 6h volume median (20-period for stability)
    vol_median_6h = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate 1d Williams Alligator (jaws, teeth, lips)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 89:  # lips period is 89
        return np.zeros(n)
    
    # Williams Alligator uses median price (hlc3)
    hlc3_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    
    # Jaws: Blue line - 13-period SMMA, shifted 8 bars ahead
    # Teeth: Red line - 8-period SMMA, shifted 5 bars ahead
    # Lips: Green line - 5-period SMMA, shifted 3 bars ahead
    # Using SMA as approximation for SMMA (simple moving average)
    jaws_1d = pd.Series(hlc3_1d).rolling(window=13, min_periods=13).mean().values
    teeth_1d = pd.Series(hlc3_1d).rolling(window=8, min_periods=8).mean().values
    lips_1d = pd.Series(hlc3_1d).rolling(window=5, min_periods=5).mean().values
    
    # Apply shifts (Williams Alligator specific)
    jaws_1d = np.concatenate([np.full(8, np.nan), jaws_1d[:-8]]) if len(jaws_1d) > 8 else np.full_like(jaws_1d, np.nan)
    teeth_1d = np.concatenate([np.full(5, np.nan), teeth_1d[:-5]]) if len(teeth_1d) > 5 else np.full_like(teeth_1d, np.nan)
    lips_1d = np.concatenate([np.full(3, np.nan), lips_1d[:-3]]) if len(lips_1d) > 3 else np.full_like(lips_1d, np.nan)
    
    # Align Alligator lines to 6h timeframe
    jaws_1d_aligned = align_htf_to_ltf(prices, df_1d, jaws_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Calculate 1d Elder Ray (Bull Power and Bear Power)
    # Bull Power = High - EMA13(close)
    # Bear Power = Low - EMA13(close)
    ema_13_1d = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = df_1d['high'].values - ema_13_1d
    bear_power_1d = df_1d['low'].values - ema_13_1d
    
    # Align Elder Ray to 6h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, volume, Alligator, and Elder Ray
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(vol_median_6h[i]) or 
            np.isnan(jaws_1d_aligned[i]) or 
            np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or 
            np.isnan(bull_power_1d_aligned[i]) or 
            np.isnan(bear_power_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Williams Alligator alignment
        bullish_alignment = (jaws_1d_aligned[i] < teeth_1d_aligned[i]) and (teeth_1d_aligned[i] < lips_1d_aligned[i])
        bearish_alignment = (jaws_1d_aligned[i] > teeth_1d_aligned[i]) and (teeth_1d_aligned[i] > lips_1d_aligned[i])
        
        # Elder Ray power
        bull_power = bull_power_1d_aligned[i]
        bear_power = bear_power_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 6h volume median
        if vol_median_6h[i] <= 0 or np.isnan(vol_median_6h[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_6h[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: bullish Alligator alignment AND Bull Power > 0 AND volume spike
            if bullish_alignment and (bull_power > 0) and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: bearish Alligator alignment AND Bear Power < 0 AND volume spike
            elif bearish_alignment and (bear_power < 0) and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Alligator alignment turns bearish OR Bull Power <= 0
            elif not bullish_alignment or (bull_power <= 0):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Alligator alignment turns bullish OR Bear Power >= 0
            elif not bearish_alignment or (bear_power >= 0):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals