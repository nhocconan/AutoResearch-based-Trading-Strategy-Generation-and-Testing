#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian channel breakout with 1-week EMA filter and volume confirmation
# Long when price breaks above 20-day Donchian upper band, above weekly EMA50, and volume > 1.5x average
# Short when price breaks below 20-day Donchian lower band, below weekly EMA50, and volume > 1.5x average
# Exit when price returns to opposite Donchian band or volume drops below average
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses daily price action with weekly trend filter to avoid counter-trend trades
# Target: 50-100 total trades over 4 years (12-25/year)

name = "1d_donchian20_weekly_ema50_vol_v1"
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
    
    # Weekly data for EMA filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA50
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Daily Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
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
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
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
            # Exit: price returns to lower Donchian band or volume drops
            elif close[i] <= donchian_low[i] or volume[i] < volume_ma[i]:
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
            # Exit: price returns to upper Donchian band or volume drops
            elif close[i] >= donchian_high[i] or volume[i] < volume_ma[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volume and weekly trend filter
            volume_surge = volume[i] > 1.5 * volume_ma[i]
            
            # Long: break above upper Donchian, above weekly EMA, volume surge
            if (close[i] > donchian_high[i] and 
                close[i] > ema_1w_aligned[i] and 
                volume_surge):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: break below lower Donchian, below weekly EMA, volume surge
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_1w_aligned[i] and 
                  volume_surge):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals