#!/usr/bin/env python3
"""
1d Donchian(20) Breakout + Weekly EMA50 Trend + Volume Spike + ATR Stoploss
Hypothesis: Daily Donchian breakouts capture strong momentum moves. Weekly EMA50 filters trend direction.
Volume spike confirms breakout strength. ATR-based stoploss manages risk. Designed for 1d timeframe
with tight entry conditions to achieve 7-25 trades/year. Works in bull (breakouts above upper band in uptrend)
and bear (breakouts below lower band in downtrend). Uses proven BTC/ETH edge from top performers.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA50 trend (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on weekly close for trend
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for stoploss and volatility filter
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    high_close[0] = high_low[0]
    low_close[0] = high_low[0]
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian, EMA, ATR, and volume MA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        upper_band = donchian_upper[i]
        lower_band = donchian_lower[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above upper Donchian band AND volume spike AND price > weekly EMA (uptrend)
            long_entry = (curr_high > upper_band) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below lower Donchian band AND volume spike AND price < weekly EMA (downtrend)
            short_entry = (curr_low < lower_band) and vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: ATR-based stoploss OR price crosses below weekly EMA (trend change)
            stop_price = entry_price - 2.5 * atr_val
            if (curr_low < stop_price) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: ATR-based stoploss OR price crosses above weekly EMA (trend change)
            stop_price = entry_price + 2.5 * atr_val
            if (curr_high > stop_price) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike_ATRStop"
timeframe = "1d"
leverage = 1.0