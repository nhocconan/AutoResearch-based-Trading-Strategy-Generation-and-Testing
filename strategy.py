#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian(20) high AND price > 12h EMA50 AND volume > 1.5x 20-period average
# Short when price breaks below Donchian(20) low AND price < 12h EMA50 AND volume > 1.5x 20-period average
# Exit when price crosses the Donchian(20) midpoint OR trend filter fails OR volume drops
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 75-200 total trades over 4 years.

name = "4h_Donchian20_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 12h calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian Channel (20) on 4h data
    period_dc = 20
    highest_high = pd.Series(high).rolling(window=period_dc, min_periods=period_dc).max().values
    lowest_low = pd.Series(low).rolling(window=period_dc, min_periods=period_dc).min().values
    dc_upper = highest_high
    dc_lower = lowest_low
    dc_middle = (dc_upper + dc_lower) / 2
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period_dc, 50, 20)  # warmup for Donchian, EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema50_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_dc_upper = dc_upper[i]
        curr_dc_lower = dc_lower[i]
        curr_dc_middle = dc_middle[i]
        curr_ema50 = ema50_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: price > Donchian upper AND price > 12h EMA50
                if curr_close > curr_dc_upper and curr_close > curr_ema50:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price < Donchian lower AND price < 12h EMA50
                elif curr_close < curr_dc_lower and curr_close < curr_ema50:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price < Donchian middle OR price < 12h EMA50 OR volume drops
            if (curr_close < curr_dc_middle) or (curr_close < curr_ema50) or (not curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price > Donchian middle OR price > 12h EMA50 OR volume drops
            if (curr_close > curr_dc_middle) or (curr_close > curr_ema50) or (not curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals