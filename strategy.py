#!/usr/bin/env python3
"""
12h_Engulfing_Signal_1wTrend_Volume
Hypothesis: 12-hour bullish/bearish engulfing candles with weekly trend filter and volume confirmation. Engulfing patterns signal strong momentum reversals, especially when aligned with the weekly trend and confirmed by volume spikes. Works in both bull and bear markets by trading with the weekly trend direction, reducing counter-trend whipsaws. Targets 15-30 trades/year by requiring pattern, trend alignment, and volume surge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA to 12h
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Trend filter: price > EMA34 = bullish, < EMA34 = bearish
    trend_up = close > ema_34_1w_aligned
    trend_down = close < ema_34_1w_aligned
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 2.0)
    
    # Bullish engulfing: current candle body > previous candle body and closes above previous open
    bullish_engulf = (close > open_) & (open_ < close) & (close > open_[1]) & (open_ < close[1]) & ((close - open_) > (open_[1] - close[1]))
    # Bearish engulfing: current candle body > previous candle body and closes below previous open
    bearish_engulf = (close < open_) & (open_ > close) & (close < open_[1]) & (open_ > close[1]) & ((open_ - close) > (close[1] - open_[1]))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_surge[i]) or 
            np.isnan(bullish_engulf[i]) or np.isnan(bearish_engulf[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with trend alignment and volume surge
        # Long: bullish engulfing + weekly uptrend + volume surge
        long_entry = bullish_engulf[i] and trend_up[i] and volume_surge[i]
        # Short: bearish engulfing + weekly downtrend + volume surge
        short_entry = bearish_engulf[i] and trend_down[i] and volume_surge[i]
        
        # Exit on opposite engulfing signal with volume surge
        long_exit = bearish_engulf[i] and volume_surge[i]
        short_exit = bullish_engulf[i] and volume_surge[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Engulfing_Signal_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0