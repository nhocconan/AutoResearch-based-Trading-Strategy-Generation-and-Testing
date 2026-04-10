#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and ATR-based stoploss
# - Long when price breaks above Donchian upper channel (20-period high) AND 12h volume > 1.3x 20-bar avg
# - Short when price breaks below Donchian lower channel (20-period low) AND 12h volume > 1.3x 20-bar avg
# - Exit when price touches Donchian middle (mean of upper/lower) or ATR stoploss hit
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - Donchian channels provide objective structure; volume confirms breakout strength
# - ATR stoploss manages risk during adverse moves

name = "4h_12h_donchian_breakout_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Pre-compute Donchian channels from 4h data (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian upper/lower/middle channels
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Pre-compute ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Pre-compute 12h volume confirmation: > 1.3x 20-period average
    volume_12h = df_12h['volume'].values
    volume_20_avg = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume_12h > (1.3 * volume_20_avg)
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(atr[i]) or
            np.isnan(vol_spike_12h_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian upper AND 12h volume spike
            if (prices['close'].iloc[i] > donchian_upper[i] and 
                vol_spike_12h_aligned[i]):
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                signals[i] = 0.25
            # Short when price breaks below Donchian lower AND 12h volume spike
            elif (prices['close'].iloc[i] < donchian_lower[i] and 
                  vol_spike_12h_aligned[i]):
                position = -1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit conditions
            # Exit conditions: 1) price touches Donchian middle, 2) ATR stoploss hit
            exit_signal = False
            if position == 1:  # Long position
                # Exit if price touches Donchian middle
                if prices['low'].iloc[i] <= donchian_middle[i]:
                    exit_signal = True
                # ATR stoploss: exit if price drops below entry - 2.0 * ATR
                elif i+1 < n and prices['low'].iloc[i] <= entry_price - 2.0 * atr[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                # Exit if price touches Donchian middle
                if prices['high'].iloc[i] >= donchian_middle[i]:
                    exit_signal = True
                # ATR stoploss: exit if price rises above entry + 2.0 * ATR
                elif i+1 < n and prices['high'].iloc[i] >= entry_price + 2.0 * atr[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals