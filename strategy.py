#!/usr/bin/env python3
"""
Hypothesis: 1d Williams %R with 1w EMA trend filter and volume confirmation.
- Primary timeframe: 1d for entries/exits.
- HTF: 1w EMA(34) for trend direction (bullish if close > EMA, bearish if close < EMA).
- Volume: Current 1d volume > 1.5 * 20-period volume MA to avoid low-volatility noise.
- Entry: Long when Williams %R(14) < -80 (oversold) AND 1w EMA trend bullish AND volume spike.
         Short when Williams %R(14) > -20 (overbought) AND 1w EMA trend bearish AND volume spike.
- Exit: Opposite Williams %R condition (%R > -50 for long exit, %R < -50 for short exit) or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
Williams %R is a momentum oscillator that identifies overbought/oversold levels.
In strong trends (filtered by 1w EMA), it can capture meaningful pullbacks for continuation entries.
Works in both bull and bear markets by aligning with the higher-timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R on 1d: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Where Highest High and Lowest Low are over the lookback period
    period14_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    period14_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((period14_high - close) / (period14_high - period14_low)) * -100
    # Avoid division by zero (when high == low)
    williams_r = np.where(period14_high == period14_low, -50, williams_r)
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA(34) for trend direction
    df_1w_close = df_1w['close'].values
    ema_34_1w = pd.Series(df_1w_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume MA on 1d for volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 1d
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    vol_ma_20_aligned = align_htf_to_ltf(prices, prices, vol_ma_20)  # 1d to 1d alignment (no change)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 34, 20)  # Need enough bars for Williams %R, 1w EMA, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        wr_val = williams_r[i]
        ema_val = ema_34_1w_aligned[i]
        vol_ma_val = vol_ma_20_aligned[i]
        
        # Volume confirmation: current 1d volume > 1.5 * 20-period volume MA
        volume_spike = volume[i] > (1.5 * vol_ma_val)
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike:
                # Bullish: Williams %R oversold (< -80) AND 1w EMA trend bullish (close > EMA)
                if wr_val < -80 and curr_close > ema_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Williams %R overbought (> -20) AND 1w EMA trend bearish (close < EMA)
                elif wr_val > -20 and curr_close < ema_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R rises above -50 (leaving oversold territory) OR loss of volume confirmation
            if wr_val > -50 or not volume_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R falls below -50 (leaving overbought territory) OR loss of volume confirmation
            if wr_val < -50 or not volume_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsR_1wEMA34_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0