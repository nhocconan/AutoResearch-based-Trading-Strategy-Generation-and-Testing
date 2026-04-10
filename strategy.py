#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme + 1d ADX trend filter + volume spike
# - Long when Williams %R(14) < -80 (oversold) + ADX(14) > 25 (trending) + 6h volume > 1.5x 20-period 6h volume SMA
# - Short when Williams %R(14) > -20 (overbought) + ADX(14) > 25 (trending) + 6h volume > 1.5x 20-period 6h volume SMA
# - Exit: Williams %R returns to -50 level (mean reversion)
# - Position sizing: 0.25 discrete level
# - Williams %R identifies overextended moves in any market regime
# - ADX filter ensures we only trade during trending periods to avoid chop
# - Volume spike confirms institutional participation in the reversal
# - Works in bull/bear: mean reversion occurs in all regimes, ADX filter prevents false signals in sideways markets

name = "6h_1d_williamsr_adx_volume_v1"
timeframe = "6h"
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
    
    # Calculate Williams %R on 6h timeframe (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Range: -100 to 0, where -80+ is oversold, -20- is overbought
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high_14 - lowest_low_14) != 0,
        (highest_high_14 - close) / (highest_high_14 - lowest_low_14) * -100,
        -50  # default to neutral when range is zero
    )
    
    # Calculate ADX on 1d timeframe (14-period)
    # ADX measures trend strength regardless of direction
    # TR = max(high-low, |high-close_prev|, |low-close_prev|)
    # +DM = max(high - high_prev, 0) if > low_prev - low else 0
    # -DM = max(low_prev - low, 0) if > high - high_prev else 0
    # DM+ = smoothed +DM, DM- = smoothed -DM
    # DX = |DM+ - DM-| / (DM+ + DM-) * 100
    # ADX = smoothed DX
    
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d.iloc[0] = df_1d['high'].iloc[0] - df_1d['low'].iloc[0]  # first bar
    
    # Directional Movement
    up_move = df_1d['high'].diff()
    down_move = -df_1d['low'].diff()  # negative because low decreases when price falls
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            # First value is simple average
            result[period-1] = np.nansum(values[:period])
            # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
            for i in range(period, len(values)):
                result[i] = result[i-1] - (result[i-1]/period) + values[i]
        return result
    
    tr_14 = wilders_smoothing(tr_1d, 14)
    plus_dm_14 = wilders_smoothing(plus_dm, 14)
    minus_dm_14 = wilders_smoothing(minus_dm, 14)
    
    # DI+ and DI-
    plus_di_14 = np.where(tr_14 != 0, plus_dm_14 / tr_14 * 100, 0)
    minus_di_14 = np.where(tr_14 != 0, minus_dm_14 / tr_14 * 100, 0)
    
    # DX and ADX
    dx_14 = np.where((plus_di_14 + minus_di_14) != 0, 
                     np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14) * 100, 
                     0)
    adx_14 = wilders_smoothing(dx_14, 14)
    
    # Align HTF indicators to 6h timeframe
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate 6h volume SMA for confirmation (20-period)
    volume_sma_20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(adx_14_aligned[i]) or np.isnan(volume_sma_20_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20_6h[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = adx_14_aligned[i] > 25
        
        # Williams %R extreme conditions
        williams_oversold = williams_r[i] < -80
        williams_overbought = williams_r[i] > -20
        
        # Entry conditions
        long_entry = williams_oversold and trend_filter and vol_confirm
        short_entry = williams_overbought and trend_filter and vol_confirm
        
        # Exit conditions: Williams %R returns to -50 level
        exit_long = williams_r[i] >= -50
        exit_short = williams_r[i] <= -50
        
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