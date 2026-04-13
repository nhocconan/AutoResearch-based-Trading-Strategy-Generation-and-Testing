#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot level touch with daily volume confirmation and ADX trend filter.
# Uses daily Camarilla pivot levels (support/resistance) from the previous day.
# Long when price touches S3/S2 with bullish momentum (ADX rising), short when touching R3/R2 with bearish momentum.
# Volume confirmation ensures institutional participation. ADX filter avoids ranging markets.
# Timeframe 12h reduces trade frequency to minimize fee drag while capturing meaningful moves.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within profitable range.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for pivot calculation (shift by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # First day will have invalid values (from roll), but will be filtered by min_periods later
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels
    s3 = prev_close - (range_val * 1.1000 / 6)
    s2 = prev_close - (range_val * 1.1000 / 4)
    s1 = prev_close - (range_val * 1.1000 / 6)
    r1 = prev_close + (range_val * 1.1000 / 6)
    r2 = prev_close + (range_val * 1.1000 / 4)
    r3 = prev_close + (range_val * 1.1000 / 6)
    
    # Calculate daily ADX for trend strength
    # True Range
    tr1 = prev_high - prev_low
    tr2 = np.abs(prev_high - np.roll(prev_close, 1))
    tr3 = np.abs(prev_low - np.roll(prev_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = np.where((prev_high - np.roll(prev_high, 1)) > (np.roll(prev_low, 1) - prev_low), 
                       np.maximum(prev_high - np.roll(prev_high, 1), 0), 0)
    minus_dm = np.where((np.roll(prev_low, 1) - prev_low) > (prev_high - np.roll(prev_high, 1)), 
                        np.maximum(np.roll(prev_low, 1) - prev_low, 0), 0)
    
    # Smooth TR, +DM, -DM
    tr_period = 14
    atr = pd.Series(tr).rolling(window=tr_period, min_periods=tr_period).mean().values
    plus_dm_smoothed = pd.Series(plus_dm).rolling(window=tr_period, min_periods=tr_period).mean().values
    minus_dm_smoothed = pd.Series(minus_dm).rolling(window=tr_period, min_periods=tr_period).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smoothed / atr
    minus_di = 100 * minus_dm_smoothed / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=tr_period, min_periods=tr_period).mean().values
    
    # Calculate daily volume and its 20-period average
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all data to 12-hour timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(s3_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 12h volume > 1.5x daily volume MA (adjusted for 12h)
        volume_12h_approx_ma = volume_ma_20_1d_aligned[i] / 2
        volume_condition = volume[i] > (volume_12h_approx_ma * 1.5)
        
        # ADX condition: trending market (ADX > 25)
        adx_condition = adx_aligned[i] > 25
        
        # Price proximity to Camarilla levels (within 0.1% of level)
        proximity_threshold = 0.001  # 0.1%
        near_s2 = abs(close[i] - s2_aligned[i]) / close[i] < proximity_threshold
        near_s3 = abs(close[i] - s3_aligned[i]) / close[i] < proximity_threshold
        near_r2 = abs(close[i] - r2_aligned[i]) / close[i] < proximity_threshold
        near_r3 = abs(close[i] - r3_aligned[i]) / close[i] < proximity_threshold
        
        # Entry conditions
        if position == 0:
            # Long when near S2/S3 with bullish momentum
            if (near_s2 or near_s3) and volume_condition and adx_condition:
                position = 1
                signals[i] = position_size
            # Short when near R2/R3 with bearish momentum
            elif (near_r2 or near_r3) and volume_condition and adx_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price moves away from support or ADX weakens
            if close[i] > s2_aligned[i] * 1.02 or adx_aligned[i] < 20:  # 2% above S2 or weak trend
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price moves away from resistance or ADX weakens
            if close[i] < r2_aligned[i] * 0.98 or adx_aligned[i] < 20:  # 2% below R2 or weak trend
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Camarilla_Pivot_Touch_Volume_ADX_Filter_v1"
timeframe = "12h"
leverage = 1.0