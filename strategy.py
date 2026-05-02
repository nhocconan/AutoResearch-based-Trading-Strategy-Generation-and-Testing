#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume spike + ATR-based trend filter
# Uses Donchian channel breakouts for trend capture with volume confirmation
# ATR-based trend filter (price > EMA20 + 0.5*ATR for longs, < for shorts) avoids choppy markets
# Volume spike (2.0x 20-period average) ensures institutional participation
# Discrete position sizing (0.25) balances profit potential with fee drag minimization
# Target: 100-180 total trades over 4 years = 25-45/year for 4h timeframe
# ATR trend filter reduces whipsaws in ranging markets while capturing strong trends

name = "4h_Donchian20_VolumeSpike_ATRTrend_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for trend filter and stoploss
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # EMA20 for dynamic trend filter
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR-based trend filter: avoids counter-trend trades in choppy markets
    # Long: price > EMA20 + 0.5*ATR, Short: price < EMA20 - 0.5*ATR
    long_filter = close > (ema20 + 0.5 * atr)
    short_filter = close < (ema20 - 0.5 * atr)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above Donchian high + volume spike + ATR trend filter
            if close[i] > highest_high[i] and volume_spike[i] and long_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low + volume spike + ATR trend filter
            elif close[i] < lowest_low[i] and volume_spike[i] and short_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below Donchian low or loss of ATR trend filter
            if close[i] < lowest_low[i] or not long_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above Donchian high or loss of ATR trend filter
            if close[i] > highest_high[i] or not short_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals