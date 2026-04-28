#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extremes with 1d EMA34 trend filter and volume spike confirmation.
# Enter long when Williams %R < -80 (oversold) with 1d EMA34 uptrend and volume > 1.8x 24-bar average.
# Enter short when Williams %R > -20 (overbought) with 1d EMA34 downtrend and volume > 1.8x 24-bar average.
# Exit when Williams %R crosses above -50 (for long) or below -50 (for short).
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 50-150 total trades over 4 years (12-37/year).
# Williams %R identifies exhaustion points; 1d EMA34 ensures higher timeframe alignment;
# volume spike filters weak reversals. Works in bull (buy oversold dips) and bear (sell overbought rallies).

name = "6h_WilliamsR_Extremes_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 6h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Williams %R (14-period) on 6h data
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: >1.8x 24-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 1.8 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Ensure sufficient history for Williams %R and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(volume_ma_24[i]) or 
            np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1d EMA34 trend: slope over 3 periods
        if i >= 3:
            ema_slope = (ema_34_aligned[i] - ema_34_aligned[i-3]) / 3
            ema_trend_up = ema_slope > 0
            ema_trend_down = ema_slope < 0
        else:
            ema_trend_up = False
            ema_trend_down = False
        
        wr = williams_r[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R < -80 (oversold), EMA34 up, volume confirm
            if wr < -80 and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R > -20 (overbought), EMA34 down, volume confirm
            elif wr > -20 and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit when Williams %R > -50
            if wr > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit when Williams %R < -50
            if wr < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals