#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h for EMA50 trend filter to capture major trend direction.
- Camarilla levels: Calculated from previous 1d OHLC (R1, S1, R3, S3, H3, L3).
- Entry: Long when close breaks above R1 AND price > 12h EMA50 AND volume > 2.0 * 20-period average volume.
         Short when close breaks below S1 AND price < 12h EMA50 AND volume > 2.0 * 20-period average volume.
- Exit: Opposite Camarilla break (close < S1 for long, close > R1 for short) OR price crosses 12h EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla levels provide intraday support/resistance that often hold in ranging markets and break in trending ones.
- 12h EMA50 provides medium-term trend filter to avoid counter-trend trades during major moves.
- Volume spike confirmation ensures breakouts have participation, reducing false signals.
- Estimated trades: ~100 total over 4 years (~25/year) based on Camarilla break frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the given OHLC."""
    # Camarilla formula based on previous day's range
    range_ = high - low
    camarilla = {
        'R4': close + range_ * 1.1 / 2,
        'R3': close + range_ * 1.1 / 4,
        'R2': close + range_ * 1.1 / 6,
        'R1': close + range_ * 1.1 / 12,
        'S1': close - range_ * 1.1 / 12,
        'S2': close - range_ * 1.1 / 6,
        'S3': close - range_ * 1.1 / 4,
        'S4': close - range_ * 1.1 / 2,
        'H3': close + range_ * 1.1 / 4,  # Same as R3
        'L3': close - range_ * 1.1 / 4,  # Same as S3
        'H4': close + range_ * 1.1 / 2,  # Same as R4
        'L4': close - range_ * 1.1 / 2,  # Same as S4
        'HH': high,
        'LL': low
    }
    return camarilla

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data for indicators
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h trend filter: EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 55:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    ema50_12h = ema(df_12h['close'].values, 50)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h, additional_delay_bars=1)
    
    # Calculate 12h volume average for confirmation
    if len(df_12h) < 21:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = df_12h['volume'].values / (vol_ma_20 + 1e-10)  # Avoid division by zero
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h, additional_delay_bars=1)
    
    # Calculate Camarilla levels from 1d OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # We need previous day's OHLC for today's Camarilla levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_r1 = np.full_like(prev_close, np.nan)
    camarilla_s1 = np.full_like(prev_close, np.nan)
    
    for i in range(len(prev_close)):
        if not (np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i])):
            camarilla = calculate_camarilla(prev_high[i], prev_low[i], prev_close[i])
            camarilla_r1[i] = camarilla['R1']
            camarilla_s1[i] = camarilla['S1']
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1, additional_delay_bars=1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for 12h EMA50 and 1d Camarilla
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ratio_12h_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_ema50 = ema50_12h_aligned[i]
        curr_r1 = camarilla_r1_aligned[i]
        curr_s1 = camarilla_s1_aligned[i]
        
        # Get 12h volume MA value for current bar (using aligned index)
        vol_ma_idx = min(i, len(vol_ma_20)-1) if len(vol_ma_20) > 0 else 0
        vol_ma_20_12h = vol_ma_20[vol_ma_idx] if len(vol_ma_20) > 0 else 0
        
        # Exit conditions: opposite Camarilla break OR price crosses 12h EMA50 in opposite direction
        if position != 0:
            # Exit long: close breaks below S1 OR price falls below 12h EMA50
            if position == 1:
                if curr_close < curr_s1 or curr_close < curr_ema50:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: close breaks above R1 OR price rises above 12h EMA50
            elif position == -1:
                if curr_close > curr_r1 or curr_close > curr_ema50:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla break with trend filter and volume confirmation
        if position == 0:
            # Volume confirmation: current volume > 2.0 * 20-period average volume
            volume_confirm = curr_volume > 2.0 * vol_ma_20_12h
            
            # Long: close breaks above R1 AND price > 12h EMA50 AND volume confirmation
            if curr_close > curr_r1 and curr_close > curr_ema50 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: close breaks below S1 AND price < 12h EMA50 AND volume confirmation
            elif curr_close < curr_s1 and curr_close < curr_ema50 and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_TrendFilter_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0