#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour 20-period Donchian breakout with weekly trend filter and volume confirmation.
# Long when: Close breaks above 6h Donchian(20) high AND weekly close > weekly open (bullish weekly candle) AND volume > 1.5 * 20-period EMA of volume.
# Short when: Close breaks below 6h Donchian(20) low AND weekly close < weekly open (bearish weekly candle) AND volume > 1.5 * 20-period EMA of volume.
# Uses Donchian for price breakout, weekly candle direction for trend filter, volume for momentum confirmation.
# Designed for low trade frequency (target: 15-25/year) to minimize fee drag and improve generalization.
# Works in bull markets via upward breakouts and in bear markets via downward breakouts.
name = "6h_Donchian20_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h Donchian Channel: 20-period high/low
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Load weekly data for trend filter (bullish/bearish weekly candle)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_open = df_1w['open'].values
    weekly_bullish = weekly_close > weekly_open  # True if bullish weekly candle
    weekly_bearish = weekly_close < weekly_open  # True if bearish weekly candle
    
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish)
    
    # Volume confirmation: current volume > 1.5 * 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high AND weekly bullish AND volume spike
            long_condition = (close[i] > donchian_high[i]) and weekly_bullish_aligned[i] and volume_spike[i]
            # Short: Break below Donchian low AND weekly bearish AND volume spike
            short_condition = (close[i] < donchian_low[i]) and weekly_bearish_aligned[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below Donchian low (reversal) or weekly turns bearish
            if close[i] < donchian_low[i] or weekly_bearish_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above Donchian high (reversal) or weekly turns bullish
            if close[i] > donchian_high[i] or weekly_bullish_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals