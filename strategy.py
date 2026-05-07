#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian channel breakout with 1-day trend filter and volume confirmation.
# Long when price breaks above upper Donchian(20) AND daily close > EMA34 (uptrend) AND volume spike.
# Short when price breaks below lower Donchian(20) AND daily close < EMA34 (downtrend) AND volume spike.
# Uses price channel breakouts for trend capture, daily EMA34 for trend filter, and volume to confirm momentum.
# Designed for moderate trade frequency (target: 25-40/year) to minimize fee drag while capturing trends.
# Works in bull markets via long breakouts in uptrend and in bear markets via short breakdowns in downtrend.
name = "4h_DonchianBreakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) on 4h data
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Donchian breakout signals: cross above upper band (long), cross below lower band (short)
    donchian_long_signal = close > highest_high_20
    donchian_short_signal = close < lowest_low_20
    
    # Load 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d trend: close > EMA34 (uptrend), close < EMA34 (downtrend)
    trend_up = close_1d > ema_34_1d
    trend_down = close_1d < ema_34_1d
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up)
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_down)
    
    # Volume confirmation: current volume > 1.5 * 20-period EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Sufficient warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian upper breakout + 1d uptrend + volume spike
            long_condition = donchian_long_signal[i] and trend_up_aligned[i] and volume_spike[i]
            # Short: Donchian lower breakout + 1d downtrend + volume spike
            short_condition = donchian_short_signal[i] and trend_down_aligned[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price closes below lower Donchian band (trend reversal) or 1d trend turns down
            if close[i] < lowest_low_20[i] or not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes above upper Donchian band (trend reversal) or 1d trend turns up
            if close[i] > highest_high_20[i] or not trend_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals