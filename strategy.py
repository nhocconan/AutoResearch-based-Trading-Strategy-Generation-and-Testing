#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 level AND 4h close > 4h EMA34 (uptrend) AND 1h volume > 1.5x 20-period volume MA.
# Short when price breaks below Camarilla S3 level AND 4h close < 4h EMA34 (downtrend) AND 1h volume > 1.5x 20-period volume MA.
# Uses session filter (08-20 UTC) to avoid low-liquidity periods. Position size fixed at 0.20.
# Designed for 1h timeframe to achieve 60-150 total trades over 4 years (15-37/year) with strict entry conditions.
# Camarilla pivots provide mathematically derived support/resistance levels, 4h EMA34 filters for trend alignment,
# volume confirmation ensures institutional participation. Works in both bull and bear markets by only trading
# breakouts in the direction of the 4h trend when volume confirms.

name = "1h_Camarilla_R3S3_4hEMA34_VolumeSpike_Session"
timeframe = "1h"
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend direction
    ema_34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 1h Camarilla pivots (based on previous day's high, low, close)
    # We need to get daily OHLC from 1h data by resampling conceptually, but we'll use rolling window approximation
    # For Camarilla, we use the previous 24-period (1 day) high, low, close
    if len(high) < 24:
        return np.zeros(n)
    
    # Calculate rolling 24-period high, low, close for pivot calculation
    prev_high = pd.Series(high).rolling(window=24, min_periods=24).max().shift(1).values
    prev_low = pd.Series(low).rolling(window=24, min_periods=24).min().shift(1).values
    prev_close = pd.Series(close).rolling(window=24, min_periods=24).mean().shift(1).values
    
    # Camarilla levels
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Volume spike detection (1h volume > 1.5x 20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(volume_ma[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Camarilla breakout conditions
        breakout_up = high_val > R3[i]  # Price breaks above Camarilla R3
        breakout_down = low_val < S3[i]  # Price breaks below Camarilla S3
        
        # 4h trend conditions
        trend_up = close_val > ema_34_4h_aligned[i]   # 4h uptrend
        trend_down = close_val < ema_34_4h_aligned[i]  # 4h downtrend
        
        if position == 0:
            # Long: Camarilla breakout up AND 4h uptrend AND volume spike AND session
            if breakout_up and trend_up and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: Camarilla breakout down AND 4h downtrend AND volume spike AND session
            elif breakout_down and trend_down and volume_spike[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price retouches Camarilla pivot point OR trend changes
            pivot_point = (prev_high[i] + prev_low[i] + prev_close[i]) / 3.0
            if close_val < pivot_point or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price retouches Camarilla pivot point OR trend changes
            pivot_point = (prev_high[i] + prev_low[i] + prev_close[i]) / 3.0
            if close_val > pivot_point or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals