#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Bands mean reversion with 12h trend filter and volume confirmation.
# Long when price touches lower BB and 12h trend is down (fade in downtrend).
# Short when price touches upper BB and 12h trend is up (fade in uptrend).
# Exit when price crosses middle band (mean reversion complete).
# Uses Bollinger Bands (20, 2) on 6h, 12h EMA(34) for trend, 60-period volume spike.
# Target: 80-120 total trades over 4 years (20-30/year) to balance opportunity and cost.

name = "6h_BB_MeanReversion_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Bollinger Bands (20, 2) on 6h
    bb_period = 20
    bb_std = 2
    close_series = pd.Series(close)
    bb_ma = close_series.rolling(window=bb_period, min_periods=bb_period).mean()
    bb_std_dev = close_series.rolling(window=bb_period, min_periods=bb_period).std()
    bb_upper = bb_ma + (bb_std_dev * bb_std)
    bb_middle = bb_ma
    bb_lower = bb_ma - (bb_std_dev * bb_std)
    
    # 12h EMA(34) for trend filter
    close_12h_series = pd.Series(close_12h)
    ema_34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = ema_34_12h[1:] > ema_34_12h[:-1]  # Rising EMA = uptrend
    trend_up = np.concatenate([[False], trend_up])  # Align with 12h index
    
    # Volume confirmation: 60-period volume spike (1.8x EMA)
    vol_ema = pd.Series(volume).ewm(span=60, adjust=False, min_periods=60).mean().values
    vol_confirm = volume > (vol_ema * 1.8)
    
    # Align indicators to 6h timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), bb_upper.values)
    bb_middle_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), bb_middle.values)
    bb_lower_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), bb_lower.values)
    trend_up_aligned = align_htf_to_ltf(prices, df_12h, trend_up.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 60)  # Ensure enough data for BB and volume EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_middle_aligned[i]) or
            np.isnan(bb_lower_aligned[i]) or np.isnan(trend_up_aligned[i]) or
            np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price at or below lower BB in downtrend (fade)
            if (trend_up_aligned[i] <= 0.5 and  # 12h downtrend
                close[i] <= bb_lower_aligned[i] and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price at or above upper BB in uptrend (fade)
            elif (trend_up_aligned[i] > 0.5 and  # 12h uptrend
                  close[i] >= bb_upper_aligned[i] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses above middle BB (mean reversion complete)
            if close[i] >= bb_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses below middle BB (mean reversion complete)
            if close[i] <= bb_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals