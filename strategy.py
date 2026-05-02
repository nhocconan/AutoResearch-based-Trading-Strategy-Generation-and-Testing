#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Uses Donchian channel breakouts (20-period high/low) for entries on daily timeframe.
# 1w EMA50 provides long-term trend filter to avoid counter-trend trades in bear markets.
# Volume confirmation (1.5x 20-period average) ensures institutional participation.
# Designed for very low trade frequency (~30-80 total trades over 4 years) to minimize fee drag.
# Works in bull markets via breakouts with trend, in bear via avoidance of false breakouts.
# Target: BTC/ETH/SOL with Sharpe > 0 on both train and test.

name = "1d_Donchian20_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for HTF calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate Donchian channels from previous 1d bar (yesterday's high/low over 20 periods)
    # We need 20 days of lookback, so we use rolling window on daily data
    # Since we don't have direct access to past daily data in 1d timeframe,
    # we calculate Donchian on the 1d timeframe itself using historical daily bars
    # For 1d timeframe, we can use rolling window directly on the price series
    donchian_h = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_l = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian, EMA, and volume MA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_aligned[i]) or np.isnan(donchian_h[i]) or 
            np.isnan(donchian_l[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian high + price > 1w EMA50 + volume confirm
            if close[i] > donchian_h[i] and close[i] > ema_50_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + price < 1w EMA50 + volume confirm
            elif close[i] < donchian_l[i] and close[i] < ema_50_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Donchian low (strong reversal signal)
            if close[i] < donchian_l[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian high (strong reversal signal)
            if close[i] > donchian_h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals