#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume spike confirmation.
# Williams Alligator uses three SMAs (jaw, teeth, lips) to identify trend and avoid chop.
# Long when lips > teeth > jaw (bullish alignment) + uptrend + volume spike.
# Short when lips < teeth < jaw (bearish alignment) + downtrend + volume spike.
# Uses ATR trailing stop (2.5x) for risk management.
# Designed for low trade frequency (~15-30/year) to minimize fee drag. Works in both bull and bear markets
# by filtering choppy markets via Alligator's "sleeping" state (intertwined lines).

name = "12h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike_ATRTrail_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: Jaw (13-period, 8-bar shift), Teeth (8-period, 5-bar shift), Lips (5-period, 3-bar shift)
    sma_13 = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    sma_8 = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    sma_5 = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    jaw = np.roll(sma_13, 8)   # shifted by 8 bars
    teeth = np.roll(sma_8, 5)  # shifted by 5 bars
    lips = np.roll(sma_5, 3)   # shifted by 3 bars
    
    # Handle NaN from rolling and roll
    jaw[:13+8] = np.nan
    teeth[:8+5] = np.nan
    lips[:5+3] = np.nan
    
    # Volume confirmation: volume > 1.8x 20-period average (tight to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    # ATR for trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 60  # warmup for Alligator (max shift 13+8=21) + EMA50
    
    for i in range(start_idx, n):
        # Skip if any Alligator value is NaN
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            signals[i] = 0.0
            continue
            
        # Regime filter: price above/below 1d EMA50 determines trend direction
        is_uptrend = close[i] > ema_50_aligned[i]
        is_downtrend = close[i] < ema_50_aligned[i]
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            if is_uptrend:
                # In uptrend: look for long when lips > teeth > jaw (bullish alignment) with volume
                if lips[i] > teeth[i] and teeth[i] > jaw[i] and curr_volume_spike:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_close
            elif is_downtrend:
                # In downtrend: look for short when lips < teeth < jaw (bearish alignment) with volume
                if lips[i] < teeth[i] and teeth[i] < jaw[i] and curr_volume_spike:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
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