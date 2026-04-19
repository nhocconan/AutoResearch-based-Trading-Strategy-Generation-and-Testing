#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly ADX filter and volume confirmation.
# Donchian breakouts capture strong directional moves. Weekly ADX > 25 filters for trending markets,
# avoiding false breakouts in ranging conditions. Volume confirmation ensures institutional participation.
# Works in bull/bear markets: ADX filter adapts to regime, breakouts capture momentum in both directions.
# Target: 10-25 trades/year per symbol.
name = "1d_Donchian20_WeeklyADX_Volume_Breakout"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for ADX filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX(14) on weekly
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # Align length
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smooth TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
        def WilderSmoothing(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            # First value is simple average
            result[period-1] = np.nanmean(data[1:period])
            # Subsequent values: Wilder smoothing
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]):
                    result[i] = result[i-1] - (result[i-1] / period) + data[i]
                else:
                    result[i] = np.nan
            return result
        
        atr = WilderSmoothing(tr, period)
        dm_plus_smooth = WilderSmoothing(dm_plus, period)
        dm_minus_smooth = WilderSmoothing(dm_minus, period)
        
        # Avoid division by zero
        dx = np.where(atr != 0, 
                      (np.abs(dm_plus_smooth - dm_minus_smooth) / atr) * 100, 
                      0)
        adx = WilderSmoothing(dx, period)
        return adx
    
    adx_14_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Donchian Channel (20) on daily
    donch_period = 20
    upper_donch = pd.Series(high).rolling(window=donch_period, min_periods=donch_period).max().values
    lower_donch = pd.Series(low).rolling(window=donch_period, min_periods=donch_period).min().values
    
    # Align weekly ADX to daily
    adx_14_aligned = align_htf_to_ltf(prices, df_1w, adx_14_1w)
    
    # Volume confirmation: current volume > 1.8x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donch_period, 14*2)  # Ensure Donchian and ADX are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_donch[i]) or np.isnan(lower_donch[i]) or 
            np.isnan(adx_14_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = upper_donch[i]
        lower = lower_donch[i]
        adx_val = adx_14_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.8 * vol_ma
        
        # Breakout conditions
        bullish_breakout = price > upper  # Price breaks above upper band
        bearish_breakout = price < lower  # Price breaks below lower band
        
        if position == 0:
            # Look for Donchian breakout in trending market (ADX > 25) with volume confirmation
            if bullish_breakout and (adx_val > 25) and volume_confirmed:
                signals[i] = 0.25
                position = 1
            elif bearish_breakout and (adx_val > 25) and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price returns to the lower Donchian band (trailing stop)
            if price < lower_donch[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price returns to the upper Donchian band
            if price > upper_donch[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals