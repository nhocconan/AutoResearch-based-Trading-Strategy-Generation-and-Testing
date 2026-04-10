#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and ATR stoploss
# - Long when price breaks above Donchian upper(20) AND 1d volume > 1.3x 20-bar avg
# - Short when price breaks below Donchian lower(20) AND 1d volume > 1.3x 20-bar avg
# - Exit when price touches Donchian middle (20-bar mean) or ATR-based stoploss
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Donchian channels provide clear trend structure; volume confirms breakout strength
# - ATR stoploss manages risk without look-ahead; middle band exit captures mean reversion

name = "4h_1d_donchian_breakout_volume_atr_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute Donchian channels from 4h data (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Donchian upper/lower: 20-period high/low
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # ATR for stoploss (20-period)
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute 1d volume confirmation: > 1.3x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.3 * volume_20_avg)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(atr[i]) or
            np.isnan(vol_spike_1d_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian upper AND 1d volume spike
            if (prices['close'].iloc[i] > donchian_upper[i] and 
                vol_spike_1d_aligned[i]):
                position = 1
                entry_price = prices['close'].iloc[i]
                atr_stop = entry_price - (2.0 * atr[i])  # 2x ATR stoploss
                signals[i] = 0.25
            # Short when price breaks below Donchian lower AND 1d volume spike
            elif (prices['close'].iloc[i] < donchian_lower[i] and 
                  vol_spike_1d_aligned[i]):
                position = -1
                entry_price = prices['close'].iloc[i]
                atr_stop = entry_price + (2.0 * atr[i])  # 2x ATR stoploss
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit conditions
            exit_signal = False
            if position == 1:  # Long position
                # Exit when price touches Donchian middle OR stoploss hit
                if (prices['close'].iloc[i] <= donchian_middle[i] or 
                    prices['close'].iloc[i] <= atr_stop):
                    exit_signal = True
            elif position == -1:  # Short position
                # Exit when price touches Donchian middle OR stoploss hit
                if (prices['close'].iloc[i] >= donchian_middle[i] or 
                    prices['close'].iloc[i] >= atr_stop):
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals