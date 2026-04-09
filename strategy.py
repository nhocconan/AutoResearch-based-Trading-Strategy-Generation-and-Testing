#!/usr/bin/env python3
# 12h_daily_higher_high_lower_low_volume_v1
# Hypothesis: 12h strategy using daily higher highs/lows structure with volume confirmation.
# Long: Daily higher high AND close above daily open, volume > 1.5x 20-period average.
# Short: Daily lower low AND close below daily open, volume > 1.5x 20-period average.
# Exit: Opposite daily structure signal or volume divergence (price move without volume support).
# Uses 1d structure for higher timeframe bias to avoid counter-trend trades.
# Volume confirmation filters weak breakouts. Target: 12-37 trades/year (50-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_daily_higher_high_lower_low_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily structure (1d HTF)
    df_1d = get_htf_data(prices, '1d')
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Daily higher high: today's high > yesterday's high
    daily_higher_high = np.zeros(len(daily_high), dtype=bool)
    daily_higher_high[1:] = daily_high[1:] > daily_high[:-1]
    
    # Daily lower low: today's low < yesterday's low
    daily_lower_low = np.zeros(len(daily_low), dtype=bool)
    daily_lower_low[1:] = daily_low[1:] < daily_low[:-1]
    
    # Daily close > open (bullish daily candle)
    daily_bullish = daily_close > daily_open
    
    # Daily close < open (bearish daily candle)
    daily_bearish = daily_close < daily_open
    
    # Align daily signals to 12h timeframe (wait for completed daily candle)
    daily_higher_high_aligned = align_htf_to_ltf(prices, df_1d, daily_higher_high.astype(float))
    daily_lower_low_aligned = align_htf_to_ltf(prices, df_1d, daily_lower_low.astype(float))
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish.astype(float))
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish.astype(float))
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(daily_higher_high_aligned[i]) or np.isnan(daily_lower_low_aligned[i]) or
            np.isnan(daily_bullish_aligned[i]) or np.isnan(daily_bearish_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(open_[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Daily structure turns bearish OR volume divergence (price up but volume down)
            if daily_bearish_aligned[i] > 0.5 or (close[i] > close[i-1] and volume[i] < volume[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Daily structure turns bullish OR volume divergence (price down but volume down)
            if daily_bullish_aligned[i] > 0.5 or (close[i] < close[i-1] and volume[i] < volume[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Daily higher high AND bullish daily candle, volume confirmed
            if (daily_higher_high_aligned[i] > 0.5 and daily_bullish_aligned[i] > 0.5 and volume_confirmed):
                position = 1
                signals[i] = 0.25
            # Short entry: Daily lower low AND bearish daily candle, volume confirmed
            elif (daily_lower_low_aligned[i] > 0.5 and daily_bearish_aligned[i] > 0.5 and volume_confirmed):
                position = -1
                signals[i] = -0.25
    
    return signals