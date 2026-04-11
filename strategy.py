#!/usr/bin/env python3
# 1d_1w_camarilla_breakout_volume_v1
# Strategy: Daily Camarilla pivot breakout with weekly trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Weekly Camarilla levels (H4/L4) act as strong support/resistance. 
# Breakouts above weekly H4 or below weekly L4 with volume confirmation and daily trend alignment 
# capture high-probability moves. Designed for very low trade frequency (~10-20/year) to minimize 
# fee drift. Works in bull markets via long breakouts and bear markets via short breakdowns.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly OHLC for Camarilla calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Weekly Camarilla levels
    # H4 = close + 1.5 * (high - low)
    # L4 = close - 1.5 * (high - low)
    weekly_range = high_1w - low_1w
    H4_1w = close_1w + 1.5 * weekly_range
    L4_1w = close_1w - 1.5 * weekly_range
    
    # Align Weekly Camarilla levels to daily timeframe
    H4_1d = align_htf_to_ltf(prices, df_1w, H4_1w)
    L4_1d = align_htf_to_ltf(prices, df_1w, L4_1w)
    
    # Daily EMA20 for trend filter
    ema_20_daily = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Daily volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if np.isnan(H4_1d[i]) or np.isnan(L4_1d[i]) or np.isnan(ema_20_daily[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume[i] > 2.0 * vol_avg_20[i]
        
        # Breakout signals
        breakout_up = high[i] > H4_1d[i-1]
        breakdown_down = low[i] < L4_1d[i-1]
        
        # Daily EMA trend filter: price above EMA = bullish trend, below = bearish
        trend_bullish = close[i] > ema_20_daily[i]
        trend_bearish = close[i] < ema_20_daily[i]
        
        # Entry conditions
        # Long: Breakout above H4 AND bullish trend AND volume confirmation
        if breakout_up and trend_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Breakdown below L4 AND bearish trend AND volume confirmation
        elif breakdown_down and trend_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite breakout (breakdown for long, breakout for short)
        elif position == 1 and breakdown_down:
            position = 0
            signals[i] = 0.0
        elif position == -1 and breakout_up:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals