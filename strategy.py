#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and ATR-based stoploss
# - Long when price breaks above Donchian upper (20) AND 1d volume > 1.8x 20-bar avg
# - Short when price breaks below Donchian lower (20) AND 1d volume > 1.8x 20-bar avg
# - Exit via ATR trailing stop: signal→0 when long price < highest_high - 2.5*ATR or short price > lowest_low + 2.5*ATR
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - Donchian channels provide clear structure; volume confirms breakout strength; ATR stop manages risk

name = "4h_1d_donchian_breakout_volume_atr_v3"
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
    
    # Pre-compute 1d volume confirmation: > 1.8x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.8 * volume_20_avg)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute ATR for trailing stop (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_spike_1d_aligned[i])):
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
            if prices['high'].iloc[i] > donchian_upper[i] and vol_spike_1d_aligned[i]:
                position = 1
                highest_since_long = prices['high'].iloc[i]
                signals[i] = 0.25
            # Short when price breaks below Donchian lower AND 1d volume spike
            elif prices['low'].iloc[i] < donchian_lower[i] and vol_spike_1d_aligned[i]:
                position = -1
                lowest_since_short = prices['low'].iloc[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for ATR trailing stop exit
            if position == 1:  # Long position
                # Update highest high since entry
                if prices['high'].iloc[i] > highest_since_long:
                    highest_since_long = prices['high'].iloc[i]
                # Check ATR trailing stop: exit if price drops below highest - 2.5*ATR
                if prices['close'].iloc[i] < highest_since_long - 2.5 * atr[i]:
                    position = 0
                    highest_since_long = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif position == -1:  # Short position
                # Update lowest low since entry
                if prices['low'].iloc[i] < lowest_since_short:
                    lowest_since_short = prices['low'].iloc[i]
                # Check ATR trailing stop: exit if price rises above lowest + 2.5*ATR
                if prices['close'].iloc[i] > lowest_since_short + 2.5 * atr[i]:
                    position = 0
                    lowest_since_short = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals