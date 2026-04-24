#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R(14) mean reversion with 12h EMA(34) trend filter and volume spike confirmation.
- Primary timeframe: 6h for entries/exits.
- HTF: 12h EMA(34) for trend direction (bullish if price > EMA34, bearish if price < EMA34).
- Volume: Current 6h volume > 2.0 * 20-period volume MA to avoid false signals.
- Entry: Long when Williams %R crosses above -80 (oversold) AND 12h EMA34 trend bullish AND volume spike.
         Short when Williams %R crosses below -20 (overbought) AND 12h EMA34 trend bearish AND volume spike.
- Exit: Opposite Williams %R cross or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Why it should work: Williams %R captures short-term extremes; 12h EMA34 ensures we trade with the higher timeframe trend; volume confirmation avoids low-liquidity false signals. Effective in both bull (buy dips) and bear (sell rallies) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period) on 6h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 12h data for EMA(34) trend and volume MA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 12h close
    ema_34 = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume MA on 12h
    vol_ma_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Volume confirmation: current 6h volume > 2.0 * 20-period 12h volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_12h_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 14, 20)  # Need enough 12h bars for EMA34 and volume MA, and 6h bars for Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_34_val = ema_34_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_williams_r = williams_r[i-1] if i > 0 else -50
        curr_williams_r = williams_r[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish entry: Williams %R crosses above -80 (from below) AND 12h EMA34 bullish (price > EMA34)
                if prev_williams_r < -80 and curr_williams_r >= -80 and curr_close > ema_34_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Williams %R crosses below -20 (from above) AND 12h EMA34 bearish (price < EMA34)
                elif prev_williams_r > -20 and curr_williams_r <= -20 and curr_close < ema_34_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 (mean reversion) OR loss of volume confirmation
            if prev_williams_r > -50 and curr_williams_r <= -50 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 (mean reversion) OR loss of volume confirmation
            if prev_williams_r < -50 and curr_williams_r >= -50 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR14_12hEMA34Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0