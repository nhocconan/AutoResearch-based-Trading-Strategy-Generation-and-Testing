#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R with 1d ADX trend filter and volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for ADX trend direction (strong trend >25) and Williams %R calculation.
- Williams %R: Measures overbought/oversold levels (-20 to -80 range).
- Entry: Long when Williams %R crosses above -80 from below AND ADX > 25 AND volume > 2.0 * 20-period average volume.
         Short when Williams %R crosses below -20 from above AND ADX > 25 AND volume > 2.0 * 20-period average volume.
- Exit: Opposite Williams %R crossover (long exits at -20, short exits at -80) OR ADX < 20 (trend weakens).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in trending markets (both bull and bear) by fading extremes only when trend is confirmed strong.
- Avoids choppy markets via ADX filter, reducing false signals and fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX) with proper min_periods."""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)
    
    for i in range(1, len(high)):
        plus_dm[i] = max(0, high[i] - high[i-1])
        minus_dm[i] = max(0, low[i-1] - low[i])
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth the values
    plus_di = 100 * ema(plus_dm, period) / ema(tr, period)
    minus_di = 100 * ema(minus_dm, period) / ema(tr, period)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = ema(dx, period)
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Williams %R and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for Williams %R and ADX
        return np.zeros(n)
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low + 1e-10)
    
    # ADX calculation
    adx_values = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Align indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 30)  # Need 20 for volume MA, 30 for Williams %R/ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Calculate Williams %R crossovers
        if i > 0:
            prev_williams = williams_r_aligned[i-1]
            curr_williams = williams_r_aligned[i]
            
            # Long entry: Williams %R crosses above -80 from below
            williams_long_cross = (prev_williams <= -80) and (curr_williams > -80)
            # Short entry: Williams %R crosses below -20 from above
            williams_short_cross = (prev_williams >= -20) and (curr_williams < -20)
            
            # Long exit: Williams %R crosses above -20 from below
            williams_long_exit = (prev_williams <= -20) and (curr_williams > -20)
            # Short exit: Williams %R crosses below -80 from above
            williams_short_exit = (prev_williams >= -80) and (curr_williams < -80)
        else:
            williams_long_cross = False
            williams_short_cross = False
            williams_long_exit = False
            williams_short_exit = False
        
        # Trend and volume filters
        strong_trend = adx_aligned[i] > 25
        weak_trend = adx_aligned[i] < 20
        volume_confirm = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Exit conditions
        if position != 0:
            # Exit long: Williams %R crosses above -20 OR trend weakens
            if position == 1:
                if williams_long_exit or weak_trend:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Williams %R crosses below -80 OR trend weakens
            elif position == -1:
                if williams_short_exit or weak_trend:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R crossover with trend and volume confirmation
        if position == 0:
            # Long: Williams %R crosses above -80 AND strong trend AND volume confirmation
            long_condition = williams_long_cross and strong_trend and volume_confirm
            
            # Short: Williams %R crosses below -20 AND strong trend AND volume confirmation
            short_condition = williams_short_cross and strong_trend and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1dADX_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0