#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Bollinger Band squeeze breakout with weekly trend filter and volume confirmation.
# Uses Bollinger Band width to identify low volatility periods (squeeze) followed by breakouts.
# Weekly EMA20 trend filter ensures alignment with higher timeframe trend.
# Volume confirmation filters out false breakouts.
# Designed to work in both bull and bear markets by capturing volatility expansions.
name = "1d_BollingerSqueeze_Breakout_WeeklyEMA20_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Bollinger Bands (20, 2)
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band width squeeze detection (lower 20% of 50-period range)
    bb_width_series = pd.Series(bb_width)
    bb_width_rank = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    squeeze_condition = bb_width_rank < 0.2  # In squeeze when BB width is in lowest 20%
    
    # Weekly data for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirmation = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(ema_20_1d[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: price breaks above upper BB, not in squeeze, with volume confirmation and above weekly EMA20
            if (price > bb_upper[i] and not squeeze_condition[i] and vol_confirmation[i] and price > ema_20_1d[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower BB, not in squeeze, with volume confirmation and below weekly EMA20
            elif (price < bb_lower[i] and not squeeze_condition[i] and vol_confirmation[i] and price < ema_20_1d[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to middle BB (mean reversion)
            if price < bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to middle BB (mean reversion)
            if price > bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals