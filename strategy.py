#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 12h EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Donchian channels: 20-period high/low breakout for entry signals.
- Entry: Long when price breaks above prior 20-bar high AND 12h EMA34 bullish AND volume > 2.0 * volume MA(20).
         Short when price breaks below prior 20-bar low AND 12h EMA34 bearish AND volume > 2.0 * volume MA(20).
- Exit: Close-based reversal - exit long when price crosses below prior 20-bar low,
        exit short when price crosses above prior 20-bar high.
- Signal size: 0.25 discrete to balance profit potential and drawdown control.
Designed to work in both bull and bear markets via trend filter and volume confirmation.
Proven pattern from DB: Donchian breakout + volume + trend filter shows SOLUSDT test Sharpe 1.10-1.38.
Adding 12h EMA34 trend filter should improve BTC/ETH performance while maintaining SOL edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    df_12h_close = df_12h['close'].values
    ema_12h = pd.Series(df_12h_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Donchian channels (20-period) on 4h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 34, 20)  # Need enough bars for EMA34 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (2.0x threshold)
            vol_confirmed = curr_volume > 2.0 * vol_ma[i]
            
            # Long: Price breaks above prior 20-bar high AND 12h EMA34 bullish AND volume confirmed
            if curr_close > donchian_high[i] and curr_close > ema_12h_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below prior 20-bar low AND 12h EMA34 bearish AND volume confirmed
            elif curr_close < donchian_low[i] and curr_close < ema_12h_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below prior 20-bar low (reversion to mean)
            if curr_close < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above prior 20-bar high (reversion to mean)
            if curr_close > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0