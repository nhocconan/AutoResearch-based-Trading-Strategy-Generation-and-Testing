#!/usr/bin/env python3
# Hypothesis: 1d Williams %R with 1w EMA200 trend filter and volume confirmation.
# Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100 over 14 periods.
# Long when %R crosses above -80 (oversold bounce) AND price > 1w EMA200 AND volume > 1.5x 20-period average volume.
# Short when %R crosses below -20 (overbought rejection) AND price < 1w EMA200 AND volume > 1.5x 20-period average volume.
# Exit when %R crosses above -20 for longs or below -80 for shorts.
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~7-25/year) by requiring confluence of mean reversion signal, weekly trend, and volume spike.
# Williams %R identifies overextended moves; EMA200 filters for major trend direction; volume confirms conviction.
# Effective in both bull and bear markets by capturing reversals at extremes with trend and volume validation.

name = "1d_Williams_%R_1wTrend_Volume_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(200) on 1w close for trend filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Williams %R (14 period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    
    # Volume confirmation: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if np.isnan(ema200_1w_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(volume_confirmation[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R crosses above -80 (from below), price > 1w EMA200, volume confirmation
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and close[i] > ema200_1w_aligned[i] and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R crosses below -20 (from above), price < 1w EMA200, volume confirmation
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and close[i] < ema200_1w_aligned[i] and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses above -20 (overbought)
            if williams_r[i] > -20 and williams_r[i-1] <= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses below -80 (oversold)
            if williams_r[i] < -80 and williams_r[i-1] >= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals