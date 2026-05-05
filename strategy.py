#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA34 trend filter + volume confirmation
# Long when price breaks above Donchian upper(20) AND close > 1w EMA34 AND volume > 2.0x 20-period average
# Short when price breaks below Donchian lower(20) AND close < 1w EMA34 AND volume > 2.0x 20-period average
# Exit when price crosses 1w EMA34 (trend reversal) OR touches opposite Donchian band
# Uses 1d primary timeframe with 1w HTF for trend filter to capture sustained moves
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag
# Donchian provides clear breakout levels; EMA34 filters for higher-timeframe trend; volume confirms participation
# Works in bull markets via breakouts and in bear markets via trend-filtered shorts

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
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA34 on 1w close for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Donchian channels from previous 20 periods (use shift to avoid look-ahead)
    prev_high = np.concatenate([[high[0]], high[:-1]])  # shift(1)
    prev_low = np.concatenate([[low[0]], low[:-1]])
    
    donchian_upper = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average (stricter for lower trade frequency)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND close > 1w EMA34 AND volume spike
            if (close[i] > donchian_upper[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND close < 1w EMA34 AND volume spike
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1w EMA34 (trend reversal) OR touches Donchian lower (support)
            if close[i] < ema_34_1w_aligned[i] or close[i] <= donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1w EMA34 (trend reversal) OR touches Donchian upper (resistance)
            if close[i] > ema_34_1w_aligned[i] or close[i] >= donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals