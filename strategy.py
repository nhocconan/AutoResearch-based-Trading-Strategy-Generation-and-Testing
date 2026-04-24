#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R with 1d EMA34 trend filter and volume spike confirmation.
- Long when Williams %R < -80 (oversold) AND 1d close > 1d EMA34 (bullish regime) AND volume > 2.0 * 20-period average volume
- Short when Williams %R > -20 (overbought) AND 1d close < 1d EMA34 (bearish regime) AND volume > 2.0 * 20-period average volume
- Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts)
- Uses 4h primary with 1d HTF to target 75-200 trades over 4 years (19-50/year)
- Williams %R identifies reversal points; EMA34 filters regime; volume spike confirms momentum
- Designed to work in both bull (buy dips) and bear (sell rallies) markets
- Signal size: 0.25 discrete levels to minimize fee churn
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
    
    # Calculate 14-period Williams %R: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # We need 14-period lookback for Williams %R
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Get daily close arrays
    daily_close = df_1d['close'].values
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Trend filter: bullish if close > EMA34, bearish if close < EMA34
    bullish_regime = close > ema_34_1d_aligned
    bearish_regime = close < ema_34_1d_aligned
    
    # Volume confirmation: volume > 2.0 * 20-period average (strong spike)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 14)  # Need EMA34 (34) and Williams %R (14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND bullish regime AND volume confirmation
            if williams_r[i] < -80 and bullish_regime[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND bearish regime AND volume confirmation
            elif williams_r[i] > -20 and bearish_regime[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -50 (moving out of oversold territory)
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -50 (moving out of overbought territory)
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0