#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 12h EMA50 trend filter + volume confirmation.
# Long when price > Alligator Teeth (Jaw) and Teeth > Lips (bullish alignment) with uptrend + volume spike.
# Short when price < Alligator Teeth and Teeth < Lips (bearish alignment) with downtrend + volume spike.
# Uses Williams Alligator (Jaw=13, Teeth=8, Lips=5) calculated on 6h close.
# ATR trailing stop (2.5x) for risk management. Targets 50-150 trades over 4 years.
# Works in both bull/bear markets by requiring 12h EMA50 trend alignment.

name = "6h_WilliamsAlligator_12hEMA50_Trend_VolumeSpike_ATRTrail_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams Alligator on 6h close: Jaw(13), Teeth(8), Lips(5)
    # All lines are smoothed with future offset (8,5,3 bars respectively) but we use aligned values
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Apply Alligator offsets: Jaw shifted by 8, Teeth by 5, Lips by 3
    jaw_offset = np.concatenate([np.full(8, np.nan), jaw[:-8]])
    teeth_offset = np.concatenate([np.full(5, np.nan), teeth[:-5]])
    lips_offset = np.concatenate([np.full(3, np.nan), lips[:-3]])
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # ATR for trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 60  # warmup for Alligator and EMA50
    
    for i in range(start_idx, n):
        # Skip if Alligator lines not available
        if np.isnan(jaw_offset[i]) or np.isnan(teeth_offset[i]) or np.isnan(lips_offset[i]):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            continue
        
        # Regime filter: price above/below 12h EMA50 determines trend direction
        is_uptrend = close[i] > ema_50_aligned[i]
        is_downtrend = close[i] < ema_50_aligned[i]
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_volume_spike = volume_spike[i]
        
        # Williams Alligator conditions
        jaw_val = jaw_offset[i]
        teeth_val = teeth_offset[i]
        lips_val = lips_offset[i]
        
        # Bullish alignment: price > teeth AND teeth > lips
        bullish_alignment = (curr_close > teeth_val) and (teeth_val > lips_val)
        # Bearish alignment: price < teeth AND teeth < lips
        bearish_alignment = (curr_close < teeth_val) and (teeth_val < lips_val)
        
        if position == 0:  # Flat - look for new entries
            if is_uptrend and bullish_alignment and curr_volume_spike:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_close
            elif is_downtrend and bearish_alignment and curr_volume_spike:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_high > highest_since_entry:
                highest_since_entry = curr_high
            
            # Trailing stop: 2.5 * ATR below highest since entry
            if curr_close < highest_since_entry - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_low < lowest_since_entry:
                lowest_since_entry = curr_low
            
            # Trailing stop: 2.5 * ATR above lowest since entry
            if curr_close > lowest_since_entry + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals