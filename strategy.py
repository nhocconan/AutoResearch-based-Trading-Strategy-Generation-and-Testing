#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian(20) breakout with 1-week EMA trend filter and volume confirmation
# Long when price breaks above Donchian upper (20) + price > weekly EMA(20) + volume > 1.5x 20-period average
# Short when price breaks below Donchian lower (20) + price < weekly EMA(20) + volume > 1.5x 20-period average
# Exit when price crosses Donchian midpoint or trend reverses
# Stoploss at 2.0 * ATR(14)
# Position size: 0.30 (30% of capital)
# Uses 1-day price channels and 1-week trend filter for multi-timeframe alignment
# Target: 30-100 total trades over 4 years (7-25/year)

name = "1d_donchian20_1w_ema_vol_v1"
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
    # Use rolling window on high/low for Donchian
    high_pd = pd.Series(high)
    low_pd = pd.Series(low)
    donchian_upper = high_pd.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_pd.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # 1-day ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.30
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses Donchian midpoint or trend reverses (price < weekly EMA)
            elif close[i] < donchian_mid[i] or close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses Donchian midpoint or trend reverses (price > weekly EMA)
            elif close[i] > donchian_mid[i] or close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.30
        else:
            # Look for entries: Donchian breakout with volume and trend filter
            # Volume filter: current volume > 1.5x 20-period average
            volume_confirmed = volume[i] > volume_threshold[i]
            
            # Long: price breaks above Donchian upper + price > weekly EMA + volume confirmed
            if close[i] > donchian_upper[i] and close[i] > ema_20_1w_aligned[i] and volume_confirmed:
                signals[i] = 0.30
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian lower + price < weekly EMA + volume confirmed
            elif close[i] < donchian_lower[i] and close[i] < ema_20_1w_aligned[i] and volume_confirmed:
                signals[i] = -0.30
                position = -1
                entry_price = close[i]
    
    return signals