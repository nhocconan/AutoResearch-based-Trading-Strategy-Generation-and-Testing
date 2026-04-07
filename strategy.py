#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day donchian breakout + 1-week ema trend + volume confirmation
# Long when price breaks above 1-day Donchian High(20) + price > 1-week EMA(20) + volume > 1.5x avg volume
# Short when price breaks below 1-day Donchian Low(20) + price < 1-week EMA(20) + volume > 1.5x avg volume
# Exit when price crosses 1-day Donchian midline or volume drops below average
# Stoploss at 2 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1-day Donchian channels for breakout and 1-week EMA for trend filter
# Target: 30-100 total trades over 4 years (7-25/year)

name = "1d_donchian20_1w_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1-week EMA(20)
    close_1w = df_1w['close'].values
    close_1w_s = pd.Series(close_1w)
    ema_20_1w = close_1w_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # 1-day Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2
    
    # Average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(avg_volume[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR
            if close[i] < entry_price - 2 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses Donchian midline or volume drops below average
            elif close[i] < donchian_mid[i] or volume[i] < avg_volume[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR
            if close[i] > entry_price + 2 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses Donchian midline or volume drops below average
            elif close[i] > donchian_mid[i] or volume[i] < avg_volume[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with EMA trend and volume confirmation
            # Long: break above Donchian high + price > weekly EMA + volume > 1.5x avg
            if close[i] > high_20[i] and close[i] > ema_20_1w_aligned[i] and volume[i] > 1.5 * avg_volume[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: break below Donchian low + price < weekly EMA + volume > 1.5x avg
            elif close[i] < low_20[i] and close[i] < ema_20_1w_aligned[i] and volume[i] > 1.5 * avg_volume[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals