#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation
# Long when Bull Power > 0, Bear Power < 0, ADX > 25 (trending), volume > 1.5x average
# Short when Bear Power < 0, Bull Power < 0, ADX > 25 (trending), volume > 1.5x average
# Exit when ADX < 20 (regime shift to ranging) or power signals reverse
# Uses discrete position sizing (0.25) to target 12-37 trades/year on 6h timeframe.
# Designed to capture trending moves in both bull and bear markets while avoiding chop.

name = "6h_ElderRay_1dADX_Regime_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Elder Ray and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for EMA13 and ADX
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema_13_1d
    bear_power = low_1d - ema_13_1d
    
    # ADX calculation (14-period)
    # +DM, -DM, TR
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(high_1d)
    tr = np.zeros_like(high_1d)
    
    for i in range(1, len(high_1d)):
        plus_dm[i] = max(0, high_1d[i] - high_1d[i-1])
        minus_dm[i] = max(0, low_1d[i-1] - low_1d[i])
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])  # First value is simple average
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period_adx = 14
    if len(plus_dm) >= period_adx:
        smoothed_plus_dm = wilder_smooth(plus_dm, period_adx)
        smoothed_minus_dm = wilder_smooth(minus_dm, period_adx)
        smoothed_tr = wilder_smooth(tr, period_adx)
        
        # Avoid division by zero
        plus_di = 100 * smoothed_plus_dm / np.where(smoothed_tr == 0, 1, smoothed_tr)
        minus_di = 100 * smoothed_minus_dm / np.where(smoothed_tr == 0, 1, smoothed_tr)
        dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1, (plus_di + minus_di))
        adx = wilder_smooth(dx, period_adx)
    else:
        adx = np.full_like(high_1d, np.nan)
    
    # Calculate 20-period average volume for confirmation (on 6h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 6h timeframe
    # Elder Ray and ADX need additional delay because they require confirmation
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power, additional_delay_bars=1)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power, additional_delay_bars=1)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Warmup for ADX and volume MA
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_bull_power = bull_power_aligned[i]
        curr_bear_power = bear_power_aligned[i]
        curr_adx = adx_aligned[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: ADX < 20 (ranging) or Bull Power <= 0 (momentum fading)
            if curr_adx < 20.0 or curr_bull_power <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ADX < 20 (ranging) or Bear Power >= 0 (momentum fading)
            if curr_adx < 20.0 or curr_bear_power >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirmed = curr_volume > 1.5 * curr_vol_ma
            
            # Long when Bull Power > 0, Bear Power < 0, ADX > 25 (strong trend), volume confirmed
            if curr_bull_power > 0 and curr_bear_power < 0 and curr_adx > 25.0 and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when Bear Power < 0, Bull Power < 0, ADX > 25 (strong trend), volume confirmed
            elif curr_bear_power < 0 and curr_bull_power < 0 and curr_adx > 25.0 and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals