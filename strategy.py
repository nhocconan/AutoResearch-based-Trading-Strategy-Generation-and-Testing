#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume spike
# Uses 4h timeframe for signal generation with Donchian channel breakouts
# 1d EMA(34) determines primary trend direction - multi-timeframe alignment with daily trend
# Volume spike (2.0x 20-period average) ensures strong institutional participation
# Discrete position sizing (0.25) minimizes fee drag while maintaining profitability
# Target: 75-200 total trades over 4 years = 19-50/year for 4h timeframe
# Donchian channels provide robust price structure based on recent highs/lows
# Works in both bull and bear markets by only taking trades aligned with 1d trend
# Prioritizes BTC/ETH over SOL by requiring volume confirmation and trend alignment

name = "4h_Donchian20_1dEMA34_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA(34) for trend determination
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels from prior 1d bar
    # Donchian(20): upper = max(high, 20), lower = min(low, 20)
    # We use prior 1d bar's high/low for breakout calculation
    prior_high = np.roll(high_1d, 1)
    prior_low = np.roll(low_1d, 1)
    prior_high[0] = np.nan  # First value has no prior
    prior_low[0] = np.nan
    
    donchian_upper = prior_high
    donchian_lower = prior_low
    
    # Align Donchian levels to 4h timeframe (they update only when new 1d bar forms)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Close > Donchian upper + volume spike + close > 1d EMA34 (bullish trend)
            if close[i] > donchian_upper_aligned[i] and volume_spike[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close < Donchian lower + volume spike + close < 1d EMA34 (bearish trend)
            elif close[i] < donchian_lower_aligned[i] and volume_spike[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close < Donchian lower or close < 1d EMA34 (trend reversal)
            if close[i] < donchian_lower_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close > Donchian upper or close > 1d EMA34 (trend reversal)
            if close[i] > donchian_upper_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals