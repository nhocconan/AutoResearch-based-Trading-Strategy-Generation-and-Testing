#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams Alligator (JAW/TEETH/LIPS) with Elder Ray (Bull/Bear Power) confirmation and volume spike
# The Alligator identifies trend phases: JAW (13-period SMMA), TEETH (8-period), LIPS (5-period). 
# When LIPS > TEETH > JAW = bullish alignment; LIPS < TEETH < JAW = bearish alignment.
# Elder Ray measures bull/bear power: Bull Power = High - EMA(13), Bear Power = Low - EMA(13).
# Long when: Alligator bullish alignment + Bull Power > 0 + volume spike
# Short when: Alligator bearish alignment + Bear Power < 0 + volume spike
# Designed for low trade frequency (<30/year) to minimize fee drag. Works in both bull/bear markets by following the Alligator's trend alignment.

name = "6h_WilliamsAlligator_ElderRay_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for Alligator and Elder Ray calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(13) for Elder Ray (Smoothed Moving Average approximation)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA(13) as proxy for SMMA(13) - close enough for trend identification
    close_1d_s = pd.Series(close_1d)
    ema_13_1d = close_1d_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power_1d = high_1d - ema_13_1d    # Bull Power = High - EMA(13)
    bear_power_1d = low_1d - ema_13_1d     # Bear Power = Low - EMA(13)
    
    # Calculate 1d Williams Alligator: JAW (13), TEETH (8), LIPS (5) - all SMMA
    # Using EMA as approximation for SMMA with same period
    jaw_1d = close_1d_s.ewm(span=13, adjust=False, min_periods=13).mean().values  # JAW
    teeth_1d = close_1d_s.ewm(span=8, adjust=False, min_periods=8).mean().values    # TEETH
    lips_1d = close_1d_s.ewm(span=5, adjust=False, min_periods=5).mean().values     # LIPS
    
    # Align all 1d indicators to 6h timeframe (wait for completed 1d bar)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Calculate ATR(14) for dynamic stoploss on 6h chart
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 30-period average
        vol_ma_30 = np.mean(volume[max(0, i-30):i]) if i >= 30 else np.mean(volume[:i]) if i > 0 else volume[i]
        volume_spike = volume[i] > (2.0 * vol_ma_30)
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_bull_power = bull_power_aligned[i]
        curr_bear_power = bear_power_aligned[i]
        curr_jaw = jaw_aligned[i]
        curr_teeth = teeth_aligned[i]
        curr_lips = lips_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Bullish entry: Alligator bullish alignment (LIPS > TEETH > JAW) + Bull Power > 0
                if curr_lips > curr_teeth and curr_teeth > curr_jaw and curr_bull_power > 0:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: Alligator bearish alignment (LIPS < TEETH < JAW) + Bear Power < 0
                elif curr_lips < curr_teeth and curr_teeth < curr_jaw and curr_bear_power < 0:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.5 * ATR below entry price OR Alligator turns bearish (LIPS < JAW)
            if curr_close < entry_price - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_lips < curr_jaw:  # Alligator sleeping/turning bearish
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.5 * ATR above entry price OR Alligator turns bullish (LIPS > JAW)
            if curr_close > entry_price + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_lips > curr_jaw:  # Alligator sleeping/turning bullish
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals