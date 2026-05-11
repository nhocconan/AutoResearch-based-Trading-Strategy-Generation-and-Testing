# 6h CCI Extreme Reversal with Volume Spike
# Hypothesis: CCI identifies overbought/oversold extremes that reverse, especially when confirmed by volume spikes.
# Works in both bull and bear markets as mean reversion at extremes. Uses 1d trend filter to avoid counter-trend trades.
# Target: 50-150 trades over 4 years via strict CCI thresholds (>250/<-250) and volume confirmation.

#!/usr/bin/env python3
name = "6h_CCI_Extreme_Reversal_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # CCI(20) calculation
    typical_price = (high + low + close) / 3
    sma_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    mad_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    # Avoid division by zero
    cci = np.where(mad_tp != 0, (typical_price - sma_tp) / (0.015 * mad_tp), 0.0)
    
    # Volume spike: current volume > 2x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (vol_ma * 2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(cci[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: CCI < -250 (oversold) AND above 1d EMA34 (uptrend) AND volume spike
            if cci[i] < -250 and close[i] > ema_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: CCI > 250 (overbought) AND below 1d EMA34 (downtrend) AND volume spike
            elif cci[i] > 250 and close[i] < ema_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: CCI crosses above -50 (reversion) OR trend breaks
            if cci[i] > -50 or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: CCI crosses below 50 (reversion) OR trend breaks
            if cci[i] < 50 or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals