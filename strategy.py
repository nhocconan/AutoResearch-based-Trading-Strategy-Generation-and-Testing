# 1h Breakout with 4h Trend Filter and Volume Confirmation
# Strategy uses 4h EMA for trend direction and 1h Donchian breakout with volume confirmation for entry.
# Works in bull markets (breakouts above Donchian in uptrend) and bear markets (breakdowns below Donchian in downtrend).
# Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag.
# Uses 4h EMA50 for trend filter and 1h Donchian(20) + volume spike for entry timing.
# Position size: 0.20 (20% of capital) to manage drawdown.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Donchian_20_4hEMA50_Volume"
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
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate EMA50 on 4h close
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_trend = ema_50_4h_aligned[i]
        upper_channel = high_max_20[i]
        lower_channel = low_min_20[i]
        
        if position == 0:
            # Long: Price breaks above upper Donchian channel AND uptrend (price > EMA50) AND volume spike
            if close_val > upper_channel and close_val > ema_trend and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below lower Donchian channel AND downtrend (price < EMA50) AND volume spike
            elif close_val < lower_channel and close_val < ema_trend and volume_spike[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks below lower Donchian channel (trend reversal) or hits opposite channel (mean reversion)
            if close_val < lower_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: Price breaks above upper Donchian channel (trend reversal) or hits opposite channel (mean reversion)
            if close_val > upper_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals