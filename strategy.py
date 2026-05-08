#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Bands squeeze + breakout with 12h volume surge and 1d trend filter.
# Bollinger Bands width < 30-day percentile indicates low volatility (squeeze).
# Breakout above upper band in uptrend (12h EMA rising) or below lower band in downtrend.
# Volume surge confirms breakout strength. Designed to catch trends after consolidation
# in both bull and bear markets with low false breakouts.

name = "6h_BollingerSqueeze_Breakout_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) on 6h
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean()
    bb_std = close_series.rolling(window=20, min_periods=20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band width percentile (30-period lookback for squeeze)
    bb_width_series = pd.Series(bb_width.values)
    bb_width_percentile = bb_width_series.rolling(window=30, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    squeeze = bb_width_percentile < 20  # Squeeze when BB width in bottom 20%
    
    # 12h EMA(34) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    close_12h_series = pd.Series(close_12h)
    ema_34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_rising = ema_34_12h[1:] > ema_34_12h[:-1]  # Rising EMA = uptrend
    ema_34_12h_rising = np.concatenate([[False], ema_34_12h_rising])  # Align with 12h index
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h_rising.astype(float))
    
    # Volume surge: current volume > 2.0x 20-period volume EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_surge = volume > (vol_ema * 2.0)
    
    # Align Bollinger Bands to 6h
    bb_upper_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), bb_upper.values)
    bb_lower_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), bb_lower.values)
    squeeze_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), squeeze.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for BB width percentile
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or
            np.isnan(squeeze_aligned[i]) or np.isnan(ema_34_12h_aligned[i]) or
            np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: breakout above upper band during squeeze release in uptrend
            if (squeeze_aligned[i] > 0.5 and  # Was in squeeze
                close[i] > bb_upper_aligned[i] and  # Break above upper band
                ema_34_12h_aligned[i] > 0.5 and  # 12h uptrend
                volume_surge[i]):  # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short entry: breakout below lower band during squeeze release in downtrend
            elif (squeeze_aligned[i] > 0.5 and  # Was in squeeze
                  close[i] < bb_lower_aligned[i] and  # Break below lower band
                  ema_34_12h_aligned[i] <= 0.5 and  # 12h downtrend
                  volume_surge[i]):  # Volume confirmation
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: reverse signal or mean reversion to middle band
            if (squeeze_aligned[i] > 0.5 and  # Back in squeeze
                close[i] < bb_middle[i]):  # Return to middle band
                signals[i] = 0.0
                position = 0
            elif (ema_34_12h_aligned[i] <= 0.5 and  # 12h trend turned down
                  close[i] < bb_lower_aligned[i]):  # Broken below lower band
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: reverse signal or mean reversion to middle band
            if (squeeze_aligned[i] > 0.5 and  # Back in squeeze
                close[i] > bb_middle[i]):  # Return to middle band
                signals[i] = 0.0
                position = 0
            elif (ema_34_12h_aligned[i] > 0.5 and  # 12h trend turned up
                  close[i] > bb_upper_aligned[i]):  # Broken above upper band
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals