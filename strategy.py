#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and ATR stoploss
# - Long when price breaks above Donchian upper channel (20-period high) AND 1d volume > 1.3x 20-bar avg
# - Short when price breaks below Donchian lower channel (20-period low) AND 1d volume > 1.3x 20-bar avg
# - Exit when price touches Donchian midpoint (mean reversion) OR ATR-based stoploss hit
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
# - Donchian channels provide clear structure; volume confirms breakout validity
# - ATR stoploss manages risk in volatile markets

name = "4h_1d_donchian_breakout_volume_atr_v1"
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
    
    # Pre-compute Donchian channels (20-period) from primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian upper channel (20-period high)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian lower channel (20-period low)
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Donchian midpoint (average of upper and lower)
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Pre-compute ATR (14-period) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 1d volume confirmation: > 1.3x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.3 * volume_20_avg)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0  # track entry price for ATR stoploss
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(atr[i]) or 
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
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                signals[i] = 0.25
            # Short when price breaks below Donchian lower AND 1d volume spike
            elif (prices['close'].iloc[i] < donchian_lower[i] and 
                  vol_spike_1d_aligned[i]):
                position = -1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit conditions
            # Exit conditions: price touches Donchian midpoint OR ATR stoploss hit
            exit_signal = False
            stop_price = 0.0
            
            if position == 1:  # Long position
                # Exit to midpoint (mean reversion)
                if prices['low'].iloc[i] <= donchian_mid[i]:
                    exit_signal = True
                # ATR stoploss: 2 * ATR below entry
                elif entry_price > 0 and prices['low'].iloc[i] <= entry_price - 2.0 * atr[i]:
                    exit_signal = True
                    stop_price = entry_price - 2.0 * atr[i]
            elif position == -1:  # Short position
                # Exit to midpoint (mean reversion)
                if prices['high'].iloc[i] >= donchian_mid[i]:
                    exit_signal = True
                # ATR stoploss: 2 * ATR above entry
                elif entry_price > 0 and prices['high'].iloc[i] >= entry_price + 2.0 * atr[i]:
                    exit_signal = True
                    stop_price = entry_price + 2.0 * atr[i]
            
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