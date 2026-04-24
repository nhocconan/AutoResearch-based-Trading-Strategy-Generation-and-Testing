#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R with 1d EMA50 trend filter and volume confirmation.
- Long when Williams %R crosses above -80 (oversold bounce) AND 1d close > 1d EMA50 (bullish regime) AND volume > 1.3 * 20-period average volume
- Short when Williams %R crosses below -20 (overbought rejection) AND 1d close < 1d EMA50 (bearish regime) AND volume > 1.3 * 20-period average volume
- Exit on opposite Williams %R level (long exits when %R crosses below -20, short exits when crosses above -80)
- Uses 12h primary with 1d HTF to target 50-150 trades over 4 years (12-37/year)
- Williams %R captures mean reversion in bear markets and momentum in bull markets; EMA50 filters regime; volume avoids fakeouts
- Designed to work in both bull (oversold bounces) and bear (overbought rejections) markets with tight entry conditions
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R calculation (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # Volume confirmation: volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma)
    
    # Get 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe (waits for completed 1d bar)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Trend filter: bullish if close > EMA50, bearish if close < EMA50
    bullish_regime = close > ema_50_1d_aligned
    bearish_regime = close < ema_50_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 20  # Need volume MA and Williams %R lookback
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 (from below) AND bullish regime AND volume confirmation
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and bullish_regime[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (from above) AND bearish regime AND volume confirmation
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and bearish_regime[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -20 (from above)
            if williams_r[i] < -20 and williams_r[i-1] >= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -80 (from below)
            if williams_r[i] > -80 and williams_r[i-1] <= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0