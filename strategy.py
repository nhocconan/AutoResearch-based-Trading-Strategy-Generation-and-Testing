#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ATR filter and volume spike confirmation
# - Long when price breaks above Donchian(20) high AND ATR(14) > 1.2x ATR(50) AND volume > 1.5x 20-bar avg
# - Short when price breaks below Donchian(20) low AND ATR(14) > 1.2x ATR(50) AND volume > 1.5x 20-bar avg
# - Exit when price touches Donchian middle (mean reversion) OR ATR volatility collapses
# - Uses volatility expansion filter to avoid choppy markets and focus on genuine breakouts
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 20-35 trades/year on 4h timeframe (80-140 total over 4 years)

name = "4h_1d_donchian_breakout_volatility_filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period)
    high_20 = prices['high'].rolling(window=20, min_periods=20).max().values
    low_20 = prices['low'].rolling(window=20, min_periods=20).min().values
    dc_middle = (high_20 + low_20) / 2
    
    # Pre-compute ATR for volatility filter
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift(1))
    low_close = np.abs(prices['low'] - prices['close'].shift(1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volatility expansion: short-term ATR > long-term ATR (expanding volatility)
    vol_expansion = atr14 > (atr50 * 1.2)
    
    # Pre-compute volume confirmation
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (volume_20_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(dc_middle[i]) or
            np.isnan(atr14[i]) or np.isnan(atr50[i]) or np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian high AND volatility expansion AND volume spike
            if (prices['close'].iloc[i] > high_20[i] and 
                vol_expansion[i] and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian low AND volatility expansion AND volume spike
            elif (prices['close'].iloc[i] < low_20[i] and 
                  vol_expansion[i] and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit conditions
            # Exit when price returns to Donchian middle OR volatility contracts
            exit_signal = False
            if position == 1:  # Long position
                if (prices['close'].iloc[i] <= dc_middle[i] or 
                    not vol_expansion[i]):
                    exit_signal = True
            elif position == -1:  # Short position
                if (prices['close'].iloc[i] >= dc_middle[i] or 
                    not vol_expansion[i]):
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