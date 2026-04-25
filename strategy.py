#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeSpike_ATRStop_v2
Hypothesis: On 4h timeframe, Donchian channel breakouts (20-period) from the previous day capture strong momentum.
Break above upper band with volume spike and 1d uptrend (price > EMA50) signals long; break below lower band with 
volume spike and 1d downtrend (price < EMA50) signals short. Uses ATR-based trailing stop and discrete position 
sizing (0.25) to limit trades (~20-40/year) and minimize fee drag. Designed for BTC/ETH to work in both bull 
and bear markets by trading breakouts with trend and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend filter (loaded ONCE)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h Donchian channel (20-period) from previous bar
    # Upper band = max(high) over last 20 periods, lower band = min(low) over last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # 4h ATR for volatility and stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    max_high = 0.0     # track highest high since entry for trailing stop (long)
    min_low = 0.0      # track lowest low since entry for trailing stop (short)
    
    # Start index: need Donchian (20), ATR (14), volume MA (20) + aligned HTF arrays
    start_idx = max(20, 14, 20, 0)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper with volume spike and 1d uptrend
            long_breakout = (curr_close > donchian_upper[i]) and vol_spike[i] and (curr_close > ema_50_1d_aligned[i])
            # Short: price breaks below Donchian lower with volume spike and 1d downtrend
            short_breakout = (curr_close < donchian_lower[i]) and vol_spike[i] and (curr_close < ema_50_1d_aligned[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                max_high = curr_high
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                min_low = curr_low
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            max_high = max(max_high, curr_high)
            # Exit: price breaks below Donchian lower OR trend turns down OR ATR trailing stop hit
            trailing_stop = curr_high < (max_high - 2.5 * atr_14[i])
            if (curr_close < donchian_lower[i]) or (curr_close < ema_50_1d_aligned[i]) or trailing_stop:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            min_low = min(min_low, curr_low)
            # Exit: price breaks above Donchian upper OR trend turns up OR ATR trailing stop hit
            trailing_stop = curr_low > (min_low + 2.5 * atr_14[i])
            if (curr_close > donchian_upper[i]) or (curr_close > ema_50_1d_aligned[i]) or trailing_stop:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeSpike_ATRStop_v2"
timeframe = "4h"
leverage = 1.0