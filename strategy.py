#!/usr/bin/env python3
"""
6h_ADX_DMI_Trend_with_12h_HTF_Volume_Confirmation
Hypothesis: On 6h timeframe, use ADX(14) > 25 and +DI > -DI for uptrend, -DI > +DI for downtrend as primary trend filter.
Confirm with 12h HTF volume spike (volume > 1.5 * 20-period average) to ensure institutional participation.
Enter long when uptrend + volume spike, short when downtrend + volume spike.
Exit when trend weakens (ADX < 20) or opposite DI crossover occurs.
Uses discrete 0.25 position size. Targets 12-25 trades/year to avoid fee drag.
Works in both bull (trend following) and bear (trend following with shorts) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === PRIMARY INDICATORS ON 6h TIMEFRAME ===
    # ADX/DMI calculation (Wilder's smoothing)
    def calculate_adx_dmi(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Wilder's smoothing (alpha = 1/period)
        def wilder_smoothing(data, period):
            result = np.full_like(data, np.nan)
            if len(data) >= period:
                # First value is simple average
                result[period-1] = np.nanmean(data[:period])
                # Subsequent values: Wilder's smoothing
                for i in range(period, len(data)):
                    result[i] = result[i-1] - (result[i-1] / period) + data[i]
            return result
        
        atr = wilder_smoothing(tr, period)
        plus_dm_smoothed = wilder_smoothing(plus_dm, period)
        minus_dm_smoothed = wilder_smoothing(minus_dm, period)
        
        # Avoid division by zero
        plus_di = np.where(atr != 0, 100 * plus_dm_smoothed / atr, 0.0)
        minus_di = np.where(atr != 0, 100 * minus_dm_smoothed / atr, 0.0)
        
        # DX and ADX
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0.0)
        adx = wilder_smoothing(dx, period)
        
        return adx, plus_di, minus_di
    
    adx, plus_di, minus_di = calculate_adx_dmi(high, low, close, 14)
    
    # === HTF VOLUME CONFIRMATION (12h) ===
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Volume spike on 12h: current volume > 1.5 * 20-period average
    vol_12h = df_12h['volume'].values
    vol_avg_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike_12h = vol_12h > (1.5 * vol_avg_12h)
    
    # Align volume spike to 6h timeframe (wait for completed 12h bar)
    volume_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 34 for ADX/DMI (2*period for Wilder smoothing stability), 20 for volume avg
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or
            np.isnan(volume_spike_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        size = 0.25  # Fixed position size
        
        if position == 0:
            # Flat - look for trend + volume confirmation
            # Long: ADX > 25, +DI > -DI (uptrend) + volume spike
            long_entry = (adx[i] > 25) and (plus_di[i] > minus_di[i]) and volume_spike_12h_aligned[i]
            # Short: ADX > 25, -DI > +DI (downtrend) + volume spike
            short_entry = (adx[i] > 25) and (minus_di[i] > plus_di[i]) and volume_spike_12h_aligned[i]
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when trend weakens or reverse crossover
            if (adx[i] < 20) or (minus_di[i] > plus_di[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when trend weakens or reverse crossover
            if (adx[i] < 20) or (plus_di[i] > minus_di[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ADX_DMI_Trend_with_12h_HTF_Volume_Confirmation"
timeframe = "6h"
leverage = 1.0