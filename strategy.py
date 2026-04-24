#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
- Uses Donchian channel (20-period high/low) from prior 1w for structure.
- Breakout above upper band or below lower band with volume confirmation captures strong moves.
- 1w EMA34 provides higher-timeframe trend filter to align with long-term momentum.
- Position size 0.25 balances profit and drawdown control.
- Target trades: 30-100 total over 4 years (7-25/year) to minimize fee drag.
- Works in bull/bear markets via 1w trend filter and volatility-based structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for Donchian channels and EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 1w EMA34 trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Donchian channels (20-period) from prior 1w candle
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Upper band: 20-period high, Lower band: 20-period low
    donchian_upper = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (wait for 1w bar to close)
    upper_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    lower_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    
    # Volume confirmation: > 1.5x 20-period average (balanced for 1d)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(upper_1w_aligned[i]) or 
            np.isnan(lower_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Only trade with volume confirmation
            if volume_confirm:
                # Long: break above upper band + above 1w EMA34 (bullish higher-timeframe trend)
                if close[i] > upper_1w_aligned[i] and close[i] > ema_34_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: break below lower band + below 1w EMA34 (bearish higher-timeframe trend)
                elif close[i] < lower_1w_aligned[i] and close[i] < ema_34_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price closes below lower band (reversal) OR below 1w EMA34 (trend change)
            if close[i] < lower_1w_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above upper band (reversal) OR above 1w EMA34 (trend change)
            if close[i] > upper_1w_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA34_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0