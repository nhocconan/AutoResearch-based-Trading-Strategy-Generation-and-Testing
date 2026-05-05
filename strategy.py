#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above 1d Donchian upper (20-period high) AND 1w close > 1w EMA34 AND volume > 2.0x 20-period average
# Short when price breaks below 1d Donchian lower (20-period low) AND 1w close < 1w EMA34 AND volume > 2.0x 20-period average
# Exit when price crosses 1w EMA34 (trend reversal) OR price retouches the 1d Donchian midpoint (mean reversion)
# Uses 1d primary timeframe with 1w HTF for trend filter (EMA34)
# Donchian channels provide clear structure for breakouts in both bull and bear markets
# Volume spike confirmation filters low-momentum false breakouts
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe

name = "1d_Donchian20_Breakout_1wEMA34_Trend_Volume"
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
    
    # Get 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d Donchian channels (20-period) - no look-ahead
    if len(high) >= 20:
        # Donchian upper: 20-period high (excluding current bar to avoid look-ahead)
        donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
        # Donchian lower: 20-period low (excluding current bar to avoid look-ahead)
        donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
        # Donchian midpoint: average of upper and lower
        donchian_mid = (donchian_upper + donchian_lower) / 2
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Volume confirmation: volume > 2.0x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND 1w close > 1w EMA34 AND volume spike
            if (close[i] > donchian_upper[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND 1w close < 1w EMA34 AND volume spike
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1w EMA34 (trend reversal) OR price retouches Donchian midpoint (mean reversion)
            if close[i] < ema_34_1w_aligned[i] or abs(close[i] - donchian_mid[i]) < 0.001 * donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1w EMA34 (trend reversal) OR price retouches Donchian midpoint (mean reversion)
            if close[i] > ema_34_1w_aligned[i] or abs(close[i] - donchian_mid[i]) < 0.001 * donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals