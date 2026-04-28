#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and ATR-based volatility filter.
# Enter long when price breaks above Donchian upper band with 1d EMA34 uptrend and ATR(14) > ATR(50) (expanding volatility).
# Enter short when price breaks below Donchian lower band with 1d EMA34 downtrend and ATR(14) > ATR(50).
# Exit when price retraces to the Donchian midpoint (average of upper and lower bands).
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 75-200 total trades over 4 years (19-50/year).
# Donchian channels provide structural breakout levels; 1d EMA34 ensures higher timeframe alignment;
# ATR expansion filter ensures breakouts occur during genuine volatility expansion, reducing false signals.
# Works in both bull (strong breakouts) and bear (strong breakdowns) by trading with the 1d trend.

name = "4h_Donchian20_Breakout_1dEMA34_ATRExpansion_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 4h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Calculate ATR(14) and ATR(50) for volatility expansion filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_expansion = atr_14 > atr_50  # Volatility expanding
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient history for ATR(50) and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(atr_50[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility expansion and volume confirmation
        vol_exp = atr_expansion[i]
        vol_confirm = volume_confirm[i]
        
        # 1d EMA34 trend: slope over 3 periods
        if i >= 3:
            ema_slope = (ema_34_aligned[i] - ema_34_aligned[i-3]) / 3
            ema_trend_up = ema_slope > 0
            ema_trend_down = ema_slope < 0
        else:
            ema_trend_up = False
            ema_trend_down = False
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Donchian upper, EMA34 up, volatility expanding, volume confirm
            if price > highest_high[i] and ema_trend_up and vol_exp and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price < Donchian lower, EMA34 down, volatility expanding, volume confirm
            elif price < lowest_low[i] and ema_trend_down and vol_exp and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit at Donchian midpoint
            if price <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit at Donchian midpoint
            if price >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals