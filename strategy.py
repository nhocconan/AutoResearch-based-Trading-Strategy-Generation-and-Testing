#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 level AND 1d close > 1d EMA34 (uptrend) AND 4h volume > 2.0x 20-period volume MA.
# Short when price breaks below Camarilla S3 level AND 1d close < 1d EMA34 (downtrend) AND 4h volume > 2.0x 20-period volume MA.
# Uses session filter (08-20 UTC) to avoid low-liquidity periods. Position size fixed at 0.25.
# Designed for 4h timeframe to achieve 75-200 total trades over 4 years (19-50/year) with strict entry conditions.
# Camarilla levels provide objective breakout zones, 1d EMA34 filters for trend alignment, 4h volume confirms participation.

name = "4h_Camarilla_R3S3_1dEMA34_VolumeSpike_Session"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend direction
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of completed daily bar)
    # We use the completed daily bar's HLC to avoid look-ahead
    # For current 4h bar, we use the previous completed 1d bar's HLC
    # Since we're using aligned data, we shift the HLC by 1 to get previous day's values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation (using previous day's HLC)
    # We need to align these to 4h and shift by 1 bar to avoid look-ahead
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    typical_price_1d_shifted = np.roll(typical_price_1d, 1)  # Previous day's typical price
    typical_price_1d_shifted[0] = np.nan  # First bar has no previous day
    
    # Calculate Camarilla levels based on previous day's range
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # But standard Camarilla uses: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    # where high, low, close are from previous day
    range_1d = high_1d - low_1d
    r3_1d = close_1d + 1.1 * range_1d
    s3_1d = close_1d - 1.1 * range_1d
    
    # Align and shift to get previous day's levels available at current 4h bar
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    # Shift by one 4h bar to ensure we're using completed previous day's levels
    r3_1d_aligned = np.roll(r3_1d_aligned, 1)
    s3_1d_aligned = np.roll(s3_1d_aligned, 1)
    r3_1d_aligned[0] = np.nan
    s3_1d_aligned[0] = np.nan
    
    # Get 4h data for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h volume 20-period MA for spike detection
    volume_ma_4h = pd.Series(df_4h['volume']).rolling(window=20, min_periods=20).mean().values
    volume_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(volume_ma_4h_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current 4h volume > 2.0x 20-period MA
        volume_spike = volume[i] > (volume_ma_4h_aligned[i] * 2.0)
        
        # 1d trend conditions
        trend_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        
        # Camarilla breakout conditions
        breakout_up = close_val > r3_1d_aligned[i]  # Price breaks above R3
        breakout_down = close_val < s3_1d_aligned[i]  # Price breaks below S3
        
        if position == 0:
            # Long: Camarilla breakout up AND 1d uptrend AND volume spike AND session
            if breakout_up and trend_up and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla breakout down AND 1d downtrend AND volume spike AND session
            elif breakout_down and trend_down and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retouches Camarilla pivot level OR trend changes
            # Pivot point = (high + low + close)/3 of previous day
            pp_1d = (high_1d + low_1d + close_1d) / 3.0
            pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
            pp_1d_aligned = np.roll(pp_1d_aligned, 1)
            pp_1d_aligned[0] = np.nan
            
            if close_val < pp_1d_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches Camarilla pivot level OR trend changes
            pp_1d = (high_1d + low_1d + close_1d) / 3.0
            pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
            pp_1d_aligned = np.roll(pp_1d_aligned, 1)
            pp_1d_aligned[0] = np.nan
            
            if close_val > pp_1d_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals