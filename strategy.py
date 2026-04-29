#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray combo with volume confirmation
# Uses Williams Alligator (JAWS/TEETH/LIPS) for trend direction and Elder Ray (Bull/Bear Power) for momentum
# Volume confirmation (>1.5x 20-period average) filters low-participation moves
# Designed for 12h timeframe to capture major swings with low frequency (target: 50-150 trades over 4 years)
# Works in both bull and bear markets: Alligator identifies trend, Elder Ray confirms momentum strength
# BTC/ETH focus: requires multi-timeframe alignment and volume confirmation to avoid noise

name = "12h_WilliamsAlligator_ElderRay_Volume_Combo"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray calculation (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate 13-period EMA for Elder Ray (using 1d close)
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate Elder Ray components (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    # Align Elder Ray components to 12h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Get 1w data for Williams Alligator (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate Williams Alligator components (using 1w median price)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    median_price_1w = (high_1w + low_1w + close_1w) / 3
    
    # Alligator JAWS (13-period SMMA, 8 bars ahead)
    jaws_1w = pd.Series(median_price_1w).rolling(window=13, min_periods=13).mean().values
    jaws_1w = np.roll(jaws_1w, 8)  # shift 8 bars ahead
    
    # Alligator TEETH (8-period SMMA, 5 bars ahead)
    teeth_1w = pd.Series(median_price_1w).rolling(window=8, min_periods=8).mean().values
    teeth_1w = np.roll(teeth_1w, 5)  # shift 5 bars ahead
    
    # Alligator LIPS (5-period SMMA, 3 bars ahead)
    lips_1w = pd.Series(median_price_1w).rolling(window=5, min_periods=5).mean().values
    lips_1w = np.roll(lips_1w, 3)  # shift 3 bars ahead
    
    # Align Alligator components to 12h timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_1w, jaws_1w)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth_1w)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips_1w)
    
    # Calculate ATR(21) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=21, min_periods=21).mean().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = max(21, 20, 13)  # ATR, volume MA, and EMA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_jaws = jaws_aligned[i]
        curr_teeth = teeth_aligned[i]
        curr_lips = lips_aligned[i]
        curr_bull_power = bull_power_aligned[i]
        curr_bear_power = bear_power_aligned[i]
        curr_atr = atr[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle stoploss and exits
        if position == 1:  # Long position
            # Stoploss: price closes below entry - 2.5 * ATR_at_entry
            if curr_close < entry_price - 2.5 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            # Exit: Alligator turns bearish (JAWS > TEETH > LIPS) or Bear Power becomes strong
            elif curr_jaws > curr_teeth > curr_lips or curr_bear_power < -0.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: price closes above entry + 2.5 * ATR_at_entry
            if curr_close > entry_price + 2.5 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            # Exit: Alligator turns bullish (LIPS > TEETH > JAWS) or Bull Power becomes strong
            elif curr_lips > curr_teeth > curr_jaws or curr_bull_power > 0.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirm = curr_volume > 1.5 * curr_vol_ma
            
            # Alligator sleeping condition: all lines intertwined (market in range)
            alligator_sleeping = (abs(curr_jaws - curr_teeth) < 0.1 * curr_atr and 
                                 abs(curr_teeth - curr_lips) < 0.1 * curr_atr and
                                 abs(curr_lips - curr_jaws) < 0.1 * curr_atr)
            
            # Long entry: Alligator awakening bullish (LIPS > TEETH > JAWS) with Bull Power confirmation
            if vol_confirm and not alligator_sleeping:
                if curr_lips > curr_teeth > curr_jaws and curr_bull_power > 0.2 * curr_atr:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
            # Short entry: Alligator awakening bearish (JAWS > TEETH > LIPS) with Bear Power confirmation
            elif vol_confirm and not alligator_sleeping:
                if curr_jaws > curr_teeth > curr_lips and curr_bear_power < -0.2 * curr_atr:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals