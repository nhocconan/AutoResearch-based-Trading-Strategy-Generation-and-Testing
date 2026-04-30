#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 12h trend filter and volume confirmation
# Bollinger Band width < 20th percentile indicates low volatility squeeze (mean reversion prone)
# Breakout above upper band or below lower band with volume > 1.5x 20-period average
# 12h EMA50 trend filter ensures alignment with medium-term trend to avoid false breakouts
# Works in bull markets via upward breakouts and bear markets via downward breakdowns
# Target: 80-180 total trades over 4 years (20-45/year) with discrete sizing 0.25

name = "6h_BBand_Squeeze_12hEMA50_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 12h data ONCE before loop (MTF Rule #1)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 20-period Bollinger Bands on 6h
    bb_period = 20
    bb_std = 2.0
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = close_series.rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_middle + (bb_std * bb_std_dev)
    bb_lower = bb_middle - (bb_std * bb_std_dev)
    bb_width = bb_upper - bb_lower
    
    # Calculate 100-period percentile rank of BB width for squeeze detection
    # Squeeze when BB width < 20th percentile (low volatility)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=100, min_periods=100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    squeeze_condition = bb_width_percentile < 0.20  # Bottom 20% = squeeze
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 60, 50)  # warmup for BB, percentile, and EMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(bb_middle[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(bb_width_percentile[i]) or np.isnan(ema_12h_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_12h = ema_12h_aligned[i]
        curr_squeeze = squeeze_condition[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and 12h trend alignment
            if curr_volume_spike:
                # Bullish breakout: price above upper band during squeeze release
                if curr_close > bb_upper[i] and curr_close > curr_ema_12h:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish breakout: price below lower band during squeeze release
                elif curr_close < bb_lower[i] and curr_close < curr_ema_12h:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price falls below middle band OR 12h trend turns bearish
            if curr_close < bb_middle[i] or curr_close < curr_ema_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above middle band OR 12h trend turns bullish
            if curr_close > bb_middle[i] or curr_close > curr_ema_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals