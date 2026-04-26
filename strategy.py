#!/usr/bin/env python3
"""
1d_WeeklyDonchian20_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: Daily Donchian(20) breakout with 1-week EMA20 trend filter and volume spike confirmation.
- Long when price breaks above daily Donchian(20) high AND 1w EMA20 uptrend AND volume > 2.0 * volume_ma(20)
- Short when price breaks below daily Donchian(20) low AND 1w EMA20 downtrend AND volume > 2.0 * volume_ma(20)
- Uses Donchian channels from completed daily bars for structure-based breakouts
- 1-week EMA20 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws in bear markets
- Volume spike (2.0x) confirms institutional participation and reduces false breakouts
- Designed for low frequency (target 7-25 trades/year) to minimize fee drag on 1d timeframe
- Exit on opposite Donchian level touch or trend reversal
- Novelty: Applies proven Donchian+HTF trend+volume pattern to 1d/1w timeframe for BTC/ETH edge in bull/bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Donchian levels (structure)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian(20) levels from prior daily bar (completed bar only)
    lookback = 20
    donch_high = pd.Series(df_1d['high'].values).rolling(window=lookback, min_periods=lookback).max().values
    donch_low = pd.Series(df_1d['low'].values).rolling(window=lookback, min_periods=lookback).min().values
    
    # Align Donchian levels to daily timeframe (no additional delay needed for structure)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Load weekly data ONCE before loop for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA20 for trend filter (needs completed 1w candle)
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    # Trend: 1 = uptrend (close > EMA20), -1 = downtrend (close < EMA20), 0 = neutral/invalid
    trend_1w = np.where(ema_20_1w_aligned > 0, 
                        np.where(close > ema_20_1w_aligned, 1, -1), 
                        0)
    
    # Calculate volume filter: volume > 2.0 * volume_ma(20) for confirmation (stricter for 1d)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for EMA, 20 for Donchian and volume MA)
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(trend_1w[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Donchian breakout conditions with trend and volume spike filter
        if position == 0:
            # Long: Price breaks above Donchian high AND 1w uptrend AND volume spike
            if close[i] > donch_high_aligned[i] and trend_1w[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND 1w downtrend AND volume spike
            elif close[i] < donch_low_aligned[i] and trend_1w[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Donchian low OR 1w trend turns down
            if close[i] < donch_low_aligned[i] or trend_1w[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Donchian high OR 1w trend turns up
            if close[i] > donch_high_aligned[i] or trend_1w[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WeeklyDonchian20_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0