#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 6h for entries/exits.
- HTF: 1d EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Volume: Current 6h volume > 1.5 * 20-period volume MA to confirm participation.
- Entry: Long when Williams %R crosses above -50 (from oversold) AND 1d EMA34 bullish AND volume spike.
         Short when Williams %R crosses below -50 (from overbought) AND 1d EMA34 bearish AND volume spike.
- Exit: Opposite Williams %R cross or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Williams %R is a momentum oscillator that identifies overbought/oversold levels.
In strong trends (EMA34 filter), pulls back to -50 offer high-probability continuation entries.
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
    
    # Calculate Williams %R on 6h (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    period14_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    period14_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (period14_high - close) / (period14_high - period14_low) * -100
    # Replace division by zero with -50 (neutral)
    williams_r = np.where((period14_high - period14_low) == 0, -50, williams_r)
    
    # Get 1d data for EMA34 and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    df_1d_close = df_1d['close'].values
    ema_34 = pd.Series(df_1d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d 20-period volume MA
    df_1d_volume = df_1d['volume'].values
    vol_ma_20 = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Volume confirmation: current 6h volume > 1.5 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (1.5 * vol_ma_20_aligned)
    
    # Williams %R signals: cross above/below -50
    # Bullish cross: previous < -50 and current >= -50
    # Bearish cross: previous > -50 and current <= -50
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = williams_r[0]  # first bar same as current
    bullish_cross = (williams_r_prev < -50) & (williams_r >= -50)
    bearish_cross = (williams_r_prev > -50) & (williams_r <= -50)
    
    # Trend filter: 1 if bullish (close > EMA34), -1 if bearish (close < EMA34), 0 otherwise
    trend_filter = np.where(close > ema_34_aligned, 1, np.where(close < ema_34_aligned, -1, 0))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 34, 20) + 1  # Need enough bars and previous value for cross
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(trend_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: Williams %R crosses above -50 AND 1d EMA34 bullish
                if bullish_cross[i] and trend_filter[i] == 1:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Williams %R crosses below -50 AND 1d EMA34 bearish
                elif bearish_cross[i] and trend_filter[i] == -1:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 OR loss of volume confirmation
            if bearish_cross[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 OR loss of volume confirmation
            if bullish_cross[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0