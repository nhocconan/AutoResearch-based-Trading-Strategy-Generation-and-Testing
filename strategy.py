#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Trend Filter with Volume and ATR Stop
# - Uses 1-week trend direction (price above/below 20-week EMA) as filter
# - Enters long when price breaks above 1-day Donchian(20) high + volume > 1.5x 20-day avg
# - Enters short when price breaks below 1-day Donchian(20) low + volume > 1.5x 20-day avg
# - Exits when price crosses back through Donchian(10) or ATR-based stop (1.5x ATR)
# - Weekly trend filter reduces whipsaw in sideways markets
# - Volume confirmation ensures breakout strength
# - Target: 15-25 trades per year per symbol (60-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 20-week EMA for trend filter
    ema_20w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20w_aligned = align_htf_to_ltf(prices, df_1w, ema_20w)
    
    # Load daily data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian channels (20 for entry, 10 for exit)
    # Donchian(20) high/low for entry signals
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Donchian(10) high/low for exit signals
    high_10 = pd.Series(high_1d).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low_1d).rolling(window=10, min_periods=10).min().values
    
    # Volume confirmation: 20-day average
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop loss (using daily data)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Align all daily indicators to 1d timeframe (no additional delay needed)
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    high_10_aligned = align_htf_to_ltf(prices, df_1d, high_10)
    low_10_aligned = align_htf_to_ltf(prices, df_1d, low_10)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Price and volume arrays
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(ema_20w_aligned[i]) or np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or \
           np.isnan(high_10_aligned[i]) or np.isnan(low_10_aligned[i]) or np.isnan(vol_ma_aligned[i]) or \
           np.isnan(atr_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian(20) high + volume surge + price above weekly EMA
            if price > high_20_aligned[i] and vol > 1.5 * vol_ma_aligned[i] and price > ema_20w_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price breaks below Donchian(20) low + volume surge + price below weekly EMA
            elif price < low_20_aligned[i] and vol > 1.5 * vol_ma_aligned[i] and price < ema_20w_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses below Donchian(10) high OR ATR stop hit (1.5*ATR)
            if price < high_10_aligned[i] or price < entry_price - 1.5 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian(10) low OR ATR stop hit (1.5*ATR)
            if price > low_10_aligned[i] or price > entry_price + 1.5 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyTrend_DonchianBreakout_Volume_ATRStop"
timeframe = "1d"
leverage = 1.0