#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout for direction, 1h for entry timing, with volume confirmation and session filter (08-20 UTC).
# Uses discrete position sizing (0.20) to minimize fee churn. Target: 60-150 total trades over 4 years (15-37/year).
# Works in bull markets via breakouts above 4h Donchian upper band, in bear markets via breakdowns below 4h Donchian lower band.
# Volume confirmation (>1.5x 20-bar MA) ensures breakouts have conviction. Session filter reduces noise during low-liquidity hours.

name = "1h_DonchianBreakout_4hDir_VolumeConfirm_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h HTF data for Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h Donchian upper and lower bands (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_20_4h_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_20_4h_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian bands to 1h timeframe (waits for completed 4h bar)
    donchian_4h_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_20_4h_high)
    donchian_4h_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_20_4h_low)
    
    # 1h volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    # Session filter: 08-20 UTC (pre-compute hour array)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for Donchian (20) and volume MA (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any indicator is NaN
        if np.isnan(donchian_4h_high_aligned[i]) or np.isnan(donchian_4h_low_aligned[i]) or np.isnan(volume_ma_20[i]):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        # Enforce session filter: only trade during 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            if position != 0:
                position = 0  # Force flat outside session
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above 4h Donchian upper band with volume confirmation
            if curr_close > donchian_4h_high_aligned[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian lower band with volume confirmation
            elif curr_close < donchian_4h_low_aligned[i] and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price retracing to 4h Donchian middle (mean of upper/lower) or opposite breakout
            donchian_mid = (donchian_4h_high_aligned[i] + donchian_4h_low_aligned[i]) / 2.0
            if curr_close < donchian_mid or curr_close < donchian_4h_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit on price retracing to 4h Donchian middle or opposite breakout
            donchian_mid = (donchian_4h_high_aligned[i] + donchian_4h_low_aligned[i]) / 2.0
            if curr_close > donchian_mid or curr_close > donchian_4h_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals