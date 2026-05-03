#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1w trend filter and volume confirmation.
# Long when price breaks above 6h Camarilla R3 level AND 1w close > 1w EMA34 (uptrend) AND 6h volume > 2.0x 20-period volume MA.
# Short when price breaks below 6h Camarilla S3 level AND 1w close < 1w EMA34 (downtrend) AND 6h volume > 2.0x 20-period volume MA.
# Exit on retracement to 6h Camarilla H3/L3 levels or trend reversal.
# Uses session filter (08-20 UTC) to avoid low-liquidity periods. Position size 0.25.
# Designed for 6h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with strict entry conditions.
# Camarilla levels provide mathematically derived support/resistance, 1w EMA34 filters for higher-timeframe trend alignment, volume confirms participation.
# Works in both bull and bear markets by only trading breakouts in the direction of the 1w trend when volume confirms.

name = "6h_Camarilla_R3S3_Breakout_1wEMA34_VolumeSpike_Session"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend direction
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for Camarilla levels (using previous 1d OHLC to avoid look-ahead)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d OHLC
    # Camarilla levels: based on previous period's range
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), R2 = close + 0.55*(high-low), R1 = close + 0.275*(high-low)
    # PP = (high+low+close)/3, S1 = close - 0.275*(high-low), S2 = close - 0.55*(high-low), S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # H3/L3 are the key levels for intraday trading
    
    # We need to shift the 1d data by 1 to avoid look-ahead (use previous period's levels)
    df_1d_shifted = df_1d.copy()
    df_1d_shifted['open'] = df_1d_shifted['open'].shift(1)
    df_1d_shifted['high'] = df_1d_shifted['high'].shift(1)
    df_1d_shifted['low'] = df_1d_shifted['low'].shift(1)
    df_1d_shifted['close'] = df_1d_shifted['close'].shift(1)
    
    # Calculate Camarilla levels using previous period's OHLC
    prev_high = df_1d_shifted['high'].values
    prev_low = df_1d_shifted['low'].values
    prev_close = df_1d_shifted['close'].values
    
    # Calculate the range
    range_hl = prev_high - prev_low
    
    # Calculate Camarilla levels
    camarilla_r3 = prev_close + 1.1 * range_hl  # R3 level
    camarilla_s3 = prev_close - 1.1 * range_hl  # S3 level
    camarilla_h3 = prev_close + 1.1 * range_hl  # H3 (same as R3 for our purposes)
    camarilla_l3 = prev_close - 1.1 * range_hl  # L3 (same as S3 for our purposes)
    camarilla_mid = (prev_high + prev_low) / 2  # Midpoint for exit
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d_shifted, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d_shifted, camarilla_s3)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d_shifted, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d_shifted, camarilla_l3)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d_shifted, camarilla_mid)
    
    # Calculate 6h volume 20-period MA for spike detection
    volume_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_mid_aligned[i]) or 
            np.isnan(volume_ma_6h[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current 6h volume > 2.0x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_6h[i] * 2.0)
        
        # Camarilla breakout conditions
        breakout_up = high_val > camarilla_r3_aligned[i]  # Price breaks above R3 level
        breakout_down = low_val < camarilla_s3_aligned[i]  # Price breaks below S3 level
        
        # 1w trend conditions
        trend_up = close_val > ema_34_1w_aligned[i]   # 1w uptrend
        trend_down = close_val < ema_34_1w_aligned[i]  # 1w downtrend
        
        if position == 0:
            # Long: Camarilla breakout up AND 1w uptrend AND volume spike AND session
            if breakout_up and trend_up and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla breakout down AND 1w downtrend AND volume spike AND session
            elif breakout_down and trend_down and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retouches Camarilla H3/L3 levels OR trend changes
            if close_val < camarilla_h3_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches Camarilla H3/L3 levels OR trend changes
            if close_val > camarilla_l3_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals