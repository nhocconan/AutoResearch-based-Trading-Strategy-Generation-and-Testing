#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
# Enter long when price breaks above 4h Donchian upper channel with 1d EMA34 uptrend and volume > 1.5x 20-bar average.
# Enter short when price breaks below 4h Donchian lower channel with 1d EMA34 downtrend and volume confirmation.
# Exit on opposite Donchian channel touch or ATR-based stoploss (2x ATR).
# Uses discrete position sizing (0.30) to balance return and drawdown.
# Donchian channels provide robust breakout levels. 1d EMA34 ensures higher timeframe trend alignment.
# Volume confirmation filters weak breakouts. ATR stoploss manages risk.
# Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 20:  # Need sufficient data for Donchian calculation
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper channel = max(high, lookback=20)
    # Donchian lower channel = min(low, lookback=20)
    # Using rolling window on 4h data
    high_series_4h = pd.Series(high_4h)
    low_series_4h = pd.Series(low_4h)
    donchian_high = high_series_4h.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series_4h.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 4h (shifted by one bar to avoid look-ahead)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:  # Need sufficient data for EMA calculation
        return np.zeros(n)
    
    # Calculate 1d EMA (34-period)
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA to 4h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate ATR (14-period) for stoploss
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 34, 14)  # Ensure sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1d EMA34 trend filter: price > EMA34 = uptrend, price < EMA34 = downtrend
        ema_trend_up = close[i] > ema_34_aligned[i]
        ema_trend_down = close[i] < ema_34_aligned[i]
        
        price = close[i]
        atr_value = atr[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Donchian upper, price > EMA34 (uptrend), volume confirm
            if price > donchian_high_aligned[i] and ema_trend_up and vol_confirm:
                signals[i] = 0.30
                position = 1
                entry_price = price
            # Short entry: price < Donchian lower, price < EMA34 (downtrend), volume confirm
            elif price < donchian_low_aligned[i] and ema_trend_down and vol_confirm:
                signals[i] = -0.30
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit conditions
            # Exit conditions: price touches Donchian lower OR stoploss hit
            if price <= donchian_low_aligned[i] or price <= entry_price - 2.0 * atr_value:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:  # Short - hold or exit conditions
            # Exit conditions: price touches Donchian upper OR stoploss hit
            if price >= donchian_high_aligned[i] or price >= entry_price + 2.0 * atr_value:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals