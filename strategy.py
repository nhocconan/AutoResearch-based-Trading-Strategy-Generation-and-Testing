# 12h_1D_VWAP_REVERSION_V1
# Hypothesis: Mean-reversion from daily VWAP with volume confirmation on 12h timeframe.
# Works in bull/bear: Price tends to revert to VWAP after deviations, especially with volume spikes.
# Uses 1d VWAP as mean reference, volume surge for conviction, and avoids overtrading with strict thresholds.
# Target: 15-25 trades/year per symbol.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1-day VWAP (typical price * volume / cumulative volume)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    cum_vol = np.cumsum(volume_1d)
    cum_tpv = np.cumsum(typical_price_1d * volume_1d)
    vwap_1d = np.where(cum_vol > 0, cum_tpv / cum_vol, np.nan)
    
    # Align 1d VWAP to 12h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate 12-period volume moving average for volume spike detection
    vol_ma = np.full_like(volume, np.nan)
    for i in range(11, len(volume)):
        vol_ma[i] = np.mean(volume[i-11:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # Conservative size to limit trades and manage drawdown
    
    for i in range(20, n):
        # Skip if critical data is unavailable
        if np.isnan(vwap_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0:
            signals[i] = 0.0
            continue
        
        volume_ratio = volume[i] / vol_ma[i]
        
        if position == 0:
            # Long: price significantly below VWAP with volume spike (oversold bounce)
            if (close[i] < 0.98 * vwap_1d_aligned[i] and  # 2% below VWAP
                volume_ratio > 2.0):                      # Strong volume confirmation
                position = 1
                signals[i] = position_size
            # Short: price significantly above VWAP with volume spike (overbought rejection)
            elif (close[i] > 1.02 * vwap_1d_aligned[i] and  # 2% above VWAP
                  volume_ratio > 2.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to VWAP (mean reversion complete)
            if close[i] >= 0.995 * vwap_1d_aligned[i]:  # Within 0.5% of VWAP
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to VWAP
            if close[i] <= 1.005 * vwap_1d_aligned[i]:  # Within 0.5% of VWAP
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1D_VWAP_REVERSION_V1"
timeframe = "12h"
leverage = 1.0