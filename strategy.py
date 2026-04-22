#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly pivot bias and volume confirmation
# Long when price breaks above Donchian(20) high, price is above weekly pivot point, and volume spike
# Short when price breaks below Donchian(20) low, price is below weekly pivot point, and volume spike
# Uses weekly pivot for trend bias to avoid counter-trend trades, works in bull/bear markets
# Target: 20-50 trades per year per symbol with disciplined entries

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for pivot points (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Donchian(20) channels on 6h data
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly pivot point calculation: (H + L + C) / 3
    pp_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    pp_1w_vals = pp_1w.values
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w_vals)
    
    # Volume spike filter (20-period on 6h data)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(pp_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout above + price above weekly PP + volume spike
            if (close[i] > donch_high[i] and 
                close[i] > pp_1w_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown below + price below weekly PP + volume spike
            elif (close[i] < donch_low[i] and 
                  close[i] < pp_1w_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Donchian reversal or loss of PP bias
            if position == 1:
                # Exit long on breakdown below Donchian low or price below weekly PP
                if close[i] < donch_low[i] or close[i] < pp_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short on breakout above Donchian high or price above weekly PP
                if close[i] > donch_high[i] or close[i] > pp_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPP_Bias_VolumeSpike"
timeframe = "6h"
leverage = 1.0