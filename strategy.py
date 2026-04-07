#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA trend filter and volume confirmation
# Long when price breaks above Donchian upper(20) and 1w EMA(20) > EMA(50) (uptrend)
# Short when price breaks below Donchian lower(20) and 1w EMA(20) < EMA(50) (downtrend)
# Exit when price crosses opposite Donchian level or stoploss at 2.0 * ATR
# Volume confirmation: current volume > 1.3 * average volume of last 20 periods
# Position size: 0.25 (25% of capital)
# Target: 50-100 total trades over 4 years (12-25/year)

name = "1d_donchian20_1w_trend_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1-week EMA(20) and EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Average volume for volume confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below Donchian lower(20)
            elif close[i] < low[i-20]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above Donchian upper(20)
            elif close[i] > high[i-20]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Calculate Donchian channels (20-period)
            highest_high = high[i-20:i].max() if i >= 20 else high[:i].max()
            lowest_low = low[i-20:i].min() if i >= 20 else low[:i].min()
            
            # Trend filter: 1w EMA(20) > EMA(50) for uptrend, < for downtrend
            uptrend = ema_20_1w_aligned[i] > ema_50_1w_aligned[i]
            downtrend = ema_20_1w_aligned[i] < ema_50_1w_aligned[i]
            
            # Volume confirmation: current volume > 1.3 * average volume
            volume_confirm = volume[i] > 1.3 * vol_avg[i]
            
            # Long: price breaks above Donchian upper(20) in uptrend with volume
            if close[i] > highest_high and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian lower(20) in downtrend with volume
            elif close[i] < lowest_low and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals