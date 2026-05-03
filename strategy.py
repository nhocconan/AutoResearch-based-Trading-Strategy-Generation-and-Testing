#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume spike.
# Long when price breaks above 1d Donchian upper (20-period high) AND 1w close > 1w EMA34 (uptrend) AND 1d volume > 2.0x 20-period volume MA.
# Short when price breaks below 1d Donchian lower (20-period low) AND 1w close < 1w EMA34 (downtrend) AND 1d volume > 2.0x 20-period volume MA.
# Exit on retracement to opposite Donchian level or trend reversal.
# Uses session filter (08-20 UTC) to avoid low-liquidity periods. Position size 0.25.
# Designed for 1d timeframe to achieve 30-100 total trades over 4 years (7-25/year) with strict entry conditions.
# Donchian channels provide robust trend-following structure, 1w EMA34 filters for higher-timeframe trend alignment, volume confirms participation.
# Works in both bull and bear markets by only trading breakouts in the direction of the 1w trend when volume confirms.

name = "1d_Donchian20_1wEMA34_VolumeSpike_Session"
timeframe = "1d"
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
    
    # Get 1d data for Donchian channels and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period high/low) from previous day to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper: 20-period high, Donchian lower: 20-period low
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (same timeframe, no shift needed)
    # But we need to shift by 1 to avoid look-ahead (use previous day's channel)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper, additional_delay_bars=1)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower, additional_delay_bars=1)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend direction
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d volume 20-period MA for spike detection
    volume_ma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(volume_ma_1d[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current 1d volume > 2.0x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_1d[i] * 2.0)
        
        # Donchian breakout conditions
        breakout_up = high_val > donchian_upper_aligned[i]   # Price breaks above upper band
        breakout_down = low_val < donchian_lower_aligned[i]  # Price breaks below lower band
        
        # 1w trend conditions
        trend_up = close_val > ema_34_1w_aligned[i]   # 1w uptrend
        trend_down = close_val < ema_34_1w_aligned[i]  # 1w downtrend
        
        if position == 0:
            # Long: Donchian breakout up AND 1w uptrend AND volume spike AND session
            if breakout_up and trend_up and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down AND 1w downtrend AND volume spike AND session
            elif breakout_down and trend_down and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retouches lower band OR trend changes
            if close_val < donchian_lower_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches upper band OR trend changes
            if close_val > donchian_upper_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals