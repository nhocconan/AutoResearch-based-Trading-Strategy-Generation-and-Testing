#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA crossover with 4h Donchian trend filter and volume confirmation
# Uses 4h Donchian(20) for higher timeframe trend direction to avoid counter-trend trades
# Uses 1h EMA(9)/EMA(21) crossover for precise entry timing within the trend
# Volume confirmation (1.5x 20-period average) ensures institutional participation
# Session filter (08-20 UTC) reduces noise during low-liquidity periods
# Discrete position sizing (0.20) balances return and risk while minimizing fee churn
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe
# Works in bull markets via trend-aligned momentum, in bear via trend filter avoiding false signals
# Combines proven EMA crossover timing with Donchian structure for robustness

name = "1h_EMA9_21_Crossover_4hDonchian20_Trend_Volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC) ONCE before loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for Donchian trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian(20) channels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe (completed 4h bar only)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate 1h EMA(9) and EMA(21) for entry timing
    close_s = pd.Series(close)
    ema_9 = close_s.ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(ema_9[i]) or np.isnan(ema_21[i]) or 
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: EMA9 > EMA21 + price > 4h Donchian high + volume confirm
            if ema_9[i] > ema_21[i] and close[i] > donchian_high_aligned[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Short: EMA9 < EMA21 + price < 4h Donchian low + volume confirm
            elif ema_9[i] < ema_21[i] and close[i] < donchian_low_aligned[i] and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: EMA9 < EMA21 or price < 4h Donchian low
            if ema_9[i] < ema_21[i] or close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: EMA9 > EMA21 or price > 4h Donchian high
            if ema_9[i] > ema_21[i] or close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals