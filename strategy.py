#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation.
# Long when close breaks above Donchian(20) high AND 12h EMA21 trend bullish AND volume > 1.3x 4h average volume.
# Short when close breaks below Donchian(20) low AND 12h EMA21 trend bearish AND volume > 1.3x 4h average volume.
# Exit when price crosses back through Donchian midpoint.
# Uses price channel breakout for entry, higher timeframe trend filter to avoid counter-trend trades,
# volume confirmation to avoid false breakouts, and midpoint exit for clean reversion.
# Target: 20-50 trades/year per symbol.
name = "4h_Donchian_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2
    
    # 12h EMA21 for trend filter
    df_12h = get_htf_data(prices, '12h')
    ema_21_12h = pd.Series(df_12h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # 4h average volume for confirmation
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 21)  # Ensure Donchian and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(ema_21_12h_aligned[i]) or np.isnan(vol_ma_4h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        upper = high_20[i]
        lower = low_20[i]
        midpoint = donchian_mid[i]
        ema_trend = ema_21_12h_aligned[i]
        vol_ma = vol_ma_4h[i]
        
        # Breakout conditions
        breakout_up = price > upper
        breakout_down = price < lower
        
        # Trend filter: price relative to 12h EMA21
        bullish_trend = price > ema_trend
        bearish_trend = price < ema_trend
        
        # Volume confirmation
        volume_confirmed = vol > 1.3 * vol_ma
        
        if position == 0:
            # Long entry: upward breakout + bullish trend + volume confirmation
            if breakout_up and bullish_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: downward breakout + bearish trend + volume confirmation
            elif breakout_down and bearish_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian midpoint
            if price < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian midpoint
            if price > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals