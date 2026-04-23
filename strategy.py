#!/usr/bin/env python3
"""
Hypothesis: 1h Donchian(20) breakout with 4h EMA50 trend filter and volume confirmation.
Long when price breaks above upper Donchian(20) AND close > 4h EMA50 AND volume > 2.0x average.
Short when price breaks below lower Donchian(20) AND close < 4h EMA50 AND volume > 2.0x average.
Exit when price crosses Donchian middle OR volume drops below 0.5x average.
Uses discrete position sizing (0.20) to minimize fee churn. Targets 20-40 trades/year per symbol.
Works in bull markets via breakouts and bear markets via short breakdowns with trend filter.
Session filter (08-20 UTC) to reduce noise trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for EMA50 trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate EMA50 on 4h data
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate Donchian channels (20-period) on 1h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_4h_aligned[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        mid = donchian_mid[i]
        price = close[i]
        vol_current = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: break above upper Donchian AND price > 4h EMA50 AND volume spike
            if (price > upper and price > ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.20
                position = 1
            # Short: break below lower Donchian AND price < 4h EMA50 AND volume spike
            elif (price < lower and price < ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses Donchian middle OR volume drops below 0.5x average
                if (price <= mid or vol_current < 0.5 * vol_ma_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses Donchian middle OR volume drops below 0.5x average
                if (price >= mid or vol_current < 0.5 * vol_ma_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Donchian20_4hEMA50_Volume_Session"
timeframe = "1h"
leverage = 1.0