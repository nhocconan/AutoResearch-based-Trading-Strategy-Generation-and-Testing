# State your hypothesis: This strategy combines 4-hour Donchian breakouts with 1-day EMA trend filter and volume confirmation.
# It aims to capture trend continuations during low volatility periods, which historically perform well in both bull and bear markets.
# The strategy uses a 20-period Donchian channel for breakouts, 1-day EMA34 for trend direction, and volume spikes for confirmation.
# It targets 20-50 trades per year on 4H timeframe to avoid excessive fee churn, with position sizing of 0.25 to manage drawdown.
# Entry requires: price breakout of Donchian channel, EMA trend alignment, and volume > 2x 20-period average.
# Exit occurs when price closes back inside the Donchian channel or crosses the EMA.
# This approach focuses on high-probability setups with clear trend alignment, reducing false signals and improving robustness.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate daily volume spike (volume > 2.0x 20-period average)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    
    # Align indicators to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_1d_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        trade_allowed = volume_spike_1d_aligned[i]
        
        if position == 0:
            # Long: Donchian breakout above upper band with EMA34 uptrend
            if trade_allowed and close[i] > donchian_high[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown below lower band with EMA34 downtrend
            elif trade_allowed and close[i] < donchian_low[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below EMA34 or Donchian lower band
            if close[i] < ema34_1d_aligned[i] or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above EMA34 or Donchian upper band
            if close[i] > ema34_1d_aligned[i] or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0