#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and ATR-based stoploss
# - Long when price breaks above Donchian upper band (20-period high) AND 12h volume > 1.3x 20-bar avg
# - Short when price breaks below Donchian lower band (20-period low) AND 12h volume > 1.3x 20-bar avg
# - Exit via ATR trailing stop: signal=0 when long and price < highest_high_since_entry - 2.5*ATR, or short and price > lowest_low_since_entry + 2.5*ATR
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Donchian channels provide objective breakout levels; volume confirms institutional participation
# - ATR stoploss adapts to volatility, reducing whipsaws in ranging markets

name = "4h_12h_donchian_breakout_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period) from 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian upper/lower bands (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR for volatility-based stoploss (14-period)
    tr1 = pd.Series(high).rolling(window=2).max() - pd.Series(low).rolling(window=2).min()
    tr2 = abs(pd.Series(high) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low) - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 12h volume confirmation: > 1.3x 20-period average
    volume_12h = df_12h['volume'].values
    volume_20_avg = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume_12h > (1.3 * volume_20_avg)
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_bar = -1  # track entry bar for highest/lowest since entry
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):  # Start after warmup for Donchian/ATR
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_spike_12h_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian upper band AND 12h volume spike
            if high[i] > donchian_upper[i] and vol_spike_12h_aligned[i]:
                position = 1
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = 0.25
            # Short when price breaks below Donchian lower band AND 12h volume spike
            elif low[i] < donchian_lower[i] and vol_spike_12h_aligned[i]:
                position = -1
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - update highest/lowest since entry and check ATR stop
            # Update highest high and lowest low since entry
            if position == 1:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                # ATR trailing stop: exit when price < highest_since_entry - 2.5*ATR
                if low[i] < (highest_since_entry - 2.5 * atr[i]):
                    position = 0
                    entry_bar = -1
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif position == -1:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                # ATR trailing stop: exit when price > lowest_since_entry + 2.5*ATR
                if high[i] > (lowest_since_entry + 2.5 * atr[i]):
                    position = 0
                    entry_bar = -1
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals