#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and 1d volume spike confirmation.
# Long when price breaks above Donchian upper AND 1d EMA34 is bullish AND 1d volume > 2.0 * 20-period average volume.
# Short when price breaks below Donchian lower AND 1d EMA34 is bearish AND 1d volume > 2.0 * 20-period average volume.
# Exit when price retraces to the Donchian midpoint (mean of upper and lower bands).
# Uses discrete position sizing (0.25) to limit fee churn. Designed for BTC/ETH robustness by capturing institutional breakouts with volume confirmation in trending markets.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.

name = "4h_Donchian20_Breakout_1dEMA34_1dVolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate EMA34
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_bullish = ema_34 > np.roll(ema_34, 1)  # Rising EMA
    ema34_bullish[0] = False  # First value has no prior
    
    # Calculate volume spike
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * vol_ma_20)  # Volume > 2x 20-period MA
    volume_spike[0:19] = False  # Not enough data for MA
    
    # Align HTF indicators to LTF
    ema34_bullish_aligned = align_htf_to_ltf(prices, df_1d, ema34_bullish.astype(float))
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Calculate Donchian channels (20-period) on 4h data
    # We need to calculate this on the LTF data directly since it's our primary timeframe
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll
    donchian_lower = low_roll
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is NaN
        if (np.isnan(ema34_bullish_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or
            np.isnan(donchian_mid[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above Donchian upper AND 1d EMA34 is bullish AND volume spike
            if (open_[i] <= donchian_upper[i] and close[i] > donchian_upper[i] and 
                ema34_bullish_aligned[i] > 0.5 and 
                volume_spike_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian lower AND 1d EMA34 is bearish AND volume spike
            elif (open_[i] >= donchian_lower[i] and close[i] < donchian_lower[i] and 
                  ema34_bullish_aligned[i] < 0.5 and 
                  volume_spike_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price retraces to Donchian midpoint
            if close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price retraces to Donchian midpoint
            if close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals