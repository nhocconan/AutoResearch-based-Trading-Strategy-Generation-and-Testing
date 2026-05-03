#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume confirmation.
# Long when price breaks above 6h Camarilla R3 level AND 12h close > 12h EMA34 (uptrend) AND 6h volume > 1.8x 20-period volume MA.
# Short when price breaks below 6h Camarilla S3 level AND 12h close < 12h EMA34 (downtrend) AND 6h volume > 1.8x 20-period volume MA.
# Exit on retracement to opposite Camarilla level (R2/S2) or trend reversal.
# Uses session filter (08-20 UTC) to avoid low-liquidity periods. Position size 0.25.
# Designed for 6h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with strict entry conditions.
# Camarilla levels provide precise intraday support/resistance, 12h EMA34 filters for higher-timeframe trend alignment, volume confirms participation.
# Works in both bull and bear markets by only trading breakouts in the direction of the 12h trend when volume confirms.

name = "6h_Camarilla_R3_S3_Breakout_12hEMA34_VolumeSpike_Session"
timeframe = "6h"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend direction
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 6h Camarilla levels from previous day (HLC of previous 12h bar)
    # Camarilla formula: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6
    #                  S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    # where C = close, H = high, L = low of previous period
    df_12h_close = df_12h['close'].values
    df_12h_high = df_12h['high'].values
    df_12h_low = df_12h['low'].values
    
    # Previous 12h bar values (shifted by 1 to avoid look-ahead)
    prev_close = np.roll(df_12h_close, 1)
    prev_high = np.roll(df_12h_high, 1)
    prev_low = np.roll(df_12h_low, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Calculate Camarilla levels for each 12h bar
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    camarilla_r2 = prev_close + (prev_high - prev_low) * 1.1 / 6
    camarilla_s2 = prev_close - (prev_high - prev_low) * 1.1 / 6
    
    # Align Camarilla levels to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    s3_6h = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    r2_6h = align_htf_to_ltf(prices, df_12h, camarilla_r2)
    s2_6h = align_htf_to_ltf(prices, df_12h, camarilla_s2)
    
    # Calculate 6h volume 20-period MA for spike detection
    volume_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA34 warmup
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(volume_ma_6h[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current 6h volume > 1.8x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_6h[i] * 1.8)
        
        # Camarilla breakout conditions
        breakout_up = high_val > r3_6h[i]   # Price breaks above R3
        breakout_down = low_val < s3_6h[i]  # Price breaks below S3
        
        # 12h trend conditions
        trend_up = close_val > ema_34_12h_aligned[i]   # 12h uptrend
        trend_down = close_val < ema_34_12h_aligned[i]  # 12h downtrend
        
        if position == 0:
            # Long: Camarilla breakout up AND 12h uptrend AND volume spike AND session
            if breakout_up and trend_up and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla breakout down AND 12h downtrend AND volume spike AND session
            elif breakout_down and trend_down and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retouches S2 level OR trend changes
            if close_val < s2_6h[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches R2 level OR trend changes
            if close_val > r2_6h[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals