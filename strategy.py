#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and ADX regime filter
# - Long when price breaks above Donchian(20) upper band + volume > 1.3x 20-period 1d volume SMA + ADX(14) > 25 (strong trend)
# - Short when price breaks below Donchian(20) lower band + volume > 1.3x 20-period 1d volume SMA + ADX(14) > 25 (strong trend)
# - Exit: price crosses Donchian(20) midline (10-period average of upper/lower bands)
# - Position sizing: 0.25 discrete level
# - Donchian breakouts capture strong momentum moves
# - Volume confirmation ensures institutional participation
# - ADX filter ensures we only trade in strong trending markets to avoid false breakouts
# - Works in bull/bear: strong trends occur in both regimes, providing breakout opportunities

name = "12h_1d_donchian_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Donchian Channels on primary timeframe (12h)
    # Upper band = 20-period high, Lower band = 20-period low
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate 1d volume SMA for confirmation (20-period)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate ADX on 1d timeframe (14-period)
    # ADX = 100 * smoothed moving average of |+DI - -DI| / (+DI + -DI)
    # +DI = 100 * smoothed moving average of +DM / TR
    # -DI = 100 * smoothed moving average of -DM / TR
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    
    # True Range components
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = df_1d['high'].iloc[0] - df_1d['low'].iloc[0]  # first bar
    
    # Directional Movement components
    up_move = df_1d['high'].diff()
    down_move = -df_1d['low'].diff()
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(values, period):
        """Wilder's smoothing: first value is SMA, subsequent values are EMA-like"""
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            # First value is simple average
            result[period-1] = np.mean(values[:period])
            # Subsequent values: smoothed = prev_smoothed * (1 - 1/period) + current_value * (1/period)
            for i in range(period, len(values)):
                result[i] = result[i-1] * (1 - 1/period) + values[i] * (1/period)
        return result
    
    atr_1d = wilders_smoothing(tr_1d, 14)
    plus_dm_smoothed = wilders_smoothing(plus_dm, 14)
    minus_dm_smoothed = wilders_smoothing(minus_dm, 14)
    
    # Calculate +DI and -DI
    plus_di_1d = np.where(atr_1d != 0, 100 * plus_dm_smoothed / atr_1d, 0.0)
    minus_di_1d = np.where(atr_1d != 0, 100 * minus_dm_smoothed / atr_1d, 0.0)
    
    # Calculate DX and ADX
    dx_1d = np.where((plus_di_1d + minus_di_1d) != 0, 
                     100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d), 
                     0.0)
    adx_1d = wilders_smoothing(dx_1d, 14)
    
    # Align ADX to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume for confirmation (aligned to 12h)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm = vol_1d_current[i] > 1.3 * volume_sma_20_1d_aligned[i]
        
        # Regime filter: ADX > 25 indicates strong trend
        regime_filter = adx_1d_aligned[i] > 25.0
        
        # Donchian breakout entry conditions
        # Long: price breaks above upper band + volume confirmation + strong trend
        # Short: price breaks below lower band + volume confirmation + strong trend
        long_entry = (close[i] > donchian_upper[i] and 
                     vol_confirm and 
                     regime_filter)
        short_entry = (close[i] < donchian_lower[i] and 
                      vol_confirm and 
                      regime_filter)
        
        # Exit conditions: price crosses Donchian midline
        exit_long = close[i] < donchian_mid[i]
        exit_short = close[i] > donchian_mid[i]
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals