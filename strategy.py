#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h trend filter and volume spike confirmation.
# Uses 4h Donchian(20) breakout for structure and 1d EMA(50) for trend filter.
# Long when price breaks above 4h upper Donchian band with bullish 1d trend and volume spike.
# Short when price breaks below 4h lower Donchian band with bearish 1d trend and volume spike.
# Target: 15-37 trades/year per symbol (60-150 total over 4 years) to minimize fee drag.
# Session filter: 08-20 UTC to avoid low-liquidity periods.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for Donchian calculation (ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian channel (20-period) on 4h data
    donchian_high = np.full(len(high_4h), np.nan)
    donchian_low = np.full(len(low_4h), np.nan)
    
    for i in range(20, len(high_4h)):
        donchian_high[i] = np.max(high_4h[i-20:i])
        donchian_low[i] = np.min(low_4h[i-20:i])
    
    # Align Donchian levels to 1h timeframe (using previous 4h bar's values)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Load 1d data for EMA(50) trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike filter (20-period on 1h data)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Higher threshold for fewer, higher quality trades
    
    # Session filter: 08-20 UTC (pre-compute before loop)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian band + bullish 1d trend + volume spike
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below lower Donchian band + bearish 1d trend + volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit on break below lower Donchian band or trend reversal
                if (close[i] < donchian_low_aligned[i] or 
                    close[i] < ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                # Exit on break above upper Donchian band or trend reversal
                if (close[i] > donchian_high_aligned[i] or 
                    close[i] > ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_1dEMA50_Trend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0