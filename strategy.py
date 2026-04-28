#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend + Volume spike + ATR-based stoploss
# Donchian channels provide robust trend-following breakouts that work in both bull/bear markets.
# 1d EMA50 ensures alignment with higher timeframe trend, reducing false signals.
# Volume confirmation filters weak breakouts.
# ATR-based stoploss manages risk by exiting when price moves against position by 2.5 ATR.
# Discrete position sizing (0.25) limits drawdown and reduces fee churn.
# Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike_v1"
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
    
    # Get 1d data for trend filter (requires daily OHLC)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for stoploss and volatility normalization
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period) for breakout signals
    # Upper channel = highest high over past 20 periods
    # Lower channel = lowest low over past 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for ATR-based stoploss
    
    start_idx = max(20, 50)  # Ensure sufficient history for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(atr[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1d EMA trend filter
        ema_trend_up = close[i] > ema_50_1d_aligned[i]
        ema_trend_down = close[i] < ema_50_1d_aligned[i]
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian upper, 1d EMA50 uptrend, volume confirm
            if price > donchian_upper[i] and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: Price breaks below Donchian lower, 1d EMA50 downtrend, volume confirm
            elif price < donchian_lower[i] and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stoploss or retracement to Donchian lower
            # ATR-based stoploss: exit if price drops below entry_price - 2.5 * ATR
            stoploss_level = entry_price - 2.5 * atr[i]
            if price < stoploss_level or price < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on stoploss or retracement to Donchian upper
            # ATR-based stoploss: exit if price rises above entry_price + 2.5 * ATR
            stoploss_level = entry_price + 2.5 * atr[i]
            if price > stoploss_level or price > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals