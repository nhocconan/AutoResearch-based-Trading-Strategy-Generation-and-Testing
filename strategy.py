#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R4/S4 breakout with 1w EMA34 trend filter and volume confirmation
# Uses tighter Camarilla levels (R4/S4) for fewer, higher-quality breakouts
# Volume confirmation > 2.0x average to filter weak breakouts
# 1w EMA34 trend filter ensures alignment with higher timeframe momentum
# Discrete position sizing (0.25) and mean reversion exit at pivot point
# Designed for 1d timeframe to target 30-100 trades over 4 years (7-25/year)
# Works in both bull and bear markets by following HTF trend and fading to pivot

name = "1d_Camarilla_R4S4_1wEMA34_VolumeSpike_v1"
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
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivot levels (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations based on previous day
    # Pivot point = (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # R4 = C + (H - L) * 1.1 / 2
    r4 = close_1d + (high_1d - low_1d) * 1.1 / 2.0
    # S4 = C - (H - L) * 1.1 / 2
    s4 = close_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # Use previous day's values (shift by 1) to avoid look-ahead
    pp_shifted = np.roll(pp, 1)
    r4_shifted = np.roll(r4, 1)
    s4_shifted = np.roll(s4, 1)
    pp_shifted[0] = np.nan
    r4_shifted[0] = np.nan
    s4_shifted[0] = np.nan
    
    # Align 1d indicators to 1d timeframe (no shift needed as already daily)
    pp_aligned = pp_shifted
    r4_aligned = r4_shifted
    s4_aligned = s4_shifted
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Volume and 1w EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_pp = pp_aligned[i]
        curr_r4 = r4_aligned[i]
        curr_s4 = s4_aligned[i]
        curr_ema34_1w = ema_34_1w_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below pivot point (mean reversion to pivot)
            if curr_close < curr_pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above pivot point (mean reversion to pivot)
            if curr_close > curr_pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average (stricter)
            vol_confirmed = curr_volume > 2.0 * curr_vol_ma
            
            # Long when price breaks above R4, 1w EMA34 up-trend, volume confirmed
            if curr_high > curr_r4 and curr_close > curr_ema34_1w and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S4, 1w EMA34 down-trend, volume confirmed
            elif curr_low < curr_s4 and curr_close < curr_ema34_1w and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals