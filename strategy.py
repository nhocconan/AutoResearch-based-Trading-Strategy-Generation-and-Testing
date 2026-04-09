#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with volume confirmation and ATR trailing stop
# - Primary timeframe: 1d, HTF: 1w for regime filter (ADX > 25 = trending)
# - Long when price closes above upper Donchian with volume > 1.5x 20-day average AND weekly ADX > 25
# - Short when price closes below lower Donchian with volume > 1.5x 20-day average AND weekly ADX > 25
# - ATR(14) trailing stop: exit long at 2.5x ATR below highest high since entry (short symmetric)
# - Fixed position size 0.25 to control drawdown
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Volume filter and ATR stop reduce false breakouts; weekly ADX ensures we only trade in trending regimes
# - Works in both bull and bear markets by capturing strong breakouts during trending phases

name = "1d_donchian_breakout_volume_adx_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 14-period ADX on weekly data
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1w[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+ and DM- using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    def WilderSmoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_1w = WilderSmoothing(tr_1w, 14)
    dm_plus_smooth = WilderSmoothing(dm_plus, 14)
    dm_minus_smooth = WilderSmoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1w != 0, 100 * dm_plus_smooth / atr_1w, 0)
    di_minus = np.where(atr_1w != 0, 100 * dm_minus_smooth / atr_1w, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = WilderSmoothing(dx, 14)
    
    # Align weekly ADX to daily timeframe (wait for completed weekly bar)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Load daily data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels on daily data
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to daily timeframe (though already daily, this ensures proper alignment)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Pre-compute volume confirmation (20-day average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute ATR (14-period) for stoploss on daily data
    tr1_d = high_1d - low_1d
    tr2_d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    tr_d[0] = tr1_d[0]
    atr_d = pd.Series(tr_d).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr_d[i]) or np.isnan(adx_aligned[i]) or
            vol_ma_20[i] <= 0 or atr_d[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Regime filter: weekly ADX > 25 indicates trending market
        is_trending = adx_aligned[i] > 25
        
        # Volume confirmation: current daily volume > 1.5x 20-day average
        volume_confirmed = volume_1d[i] > 1.5 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high_1d[i])
            
            # ATR-based trailing stop: exit if price drops 2.5x ATR from highest high
            if close_1d[i] < highest_high_since_entry - 2.5 * atr_d[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low_1d[i])
            
            # ATR-based trailing stop: exit if price rises 2.5x ATR from lowest low
            if close_1d[i] > lowest_low_since_entry + 2.5 * atr_d[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Donchian breakout + volume confirmation + trend regime
            if volume_confirmed and is_trending:
                # Long entry: price closes above upper Donchian
                if close_1d[i] > upper_aligned[i]:
                    position = 1
                    highest_high_since_entry = high_1d[i]
                    lowest_low_since_entry = low_1d[i]
                    signals[i] = 0.25
                # Short entry: price closes below lower Donchian
                elif close_1d[i] < lower_aligned[i]:
                    position = -1
                    highest_high_since_entry = high_1d[i]
                    lowest_low_since_entry = low_1d[i]
                    signals[i] = -0.25
    
    return signals