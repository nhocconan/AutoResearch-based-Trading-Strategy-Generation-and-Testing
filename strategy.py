#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout + 1-week EMA(50) trend filter + volume confirmation
# Long when price breaks above Donchian upper band with volume > 2x average and close > weekly EMA50
# Short when price breaks below Donchian lower band with volume > 2x average and close < weekly EMA50
# Exit when price crosses below/above weekly EMA50 or Donchian middle band
# Stoploss at 2.5 * ATR(20)
# Position size: 0.30 (30% of capital)
# Target: 40-100 total trades over 4 years (10-25/year)

name = "1d_donchian20_weekly_ema50_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for EMA50 trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Weekly EMA50
    close_weekly = df_weekly['close'].values
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Daily Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Daily volume average (20-period)
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # ATR(20) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(ema50_weekly_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.30
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below weekly EMA50 or Donchian middle
            elif close[i] < ema50_weekly_aligned[i] or close[i] < donchian_middle[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above weekly EMA50 or Donchian middle
            elif close[i] > ema50_weekly_aligned[i] or close[i] > donchian_middle[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.30
        else:
            # Look for entries: Donchian breakout with volume spike and weekly EMA50 filter
            volume_spike = volume[i] > 2.0 * volume_ma[i]
            
            # Long: break above Donchian upper with volume spike and above weekly EMA50
            if (close[i] > donchian_upper[i] and 
                volume_spike and 
                close[i] > ema50_weekly_aligned[i]):
                signals[i] = 0.30
                position = 1
                entry_price = close[i]
            # Short: break below Donchian lower with volume spike and below weekly EMA50
            elif (close[i] < donchian_lower[i] and 
                  volume_spike and 
                  close[i] < ema50_weekly_aligned[i]):
                signals[i] = -0.30
                position = -1
                entry_price = close[i]
    
    return signals