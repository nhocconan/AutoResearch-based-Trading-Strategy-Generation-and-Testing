#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above upper BB after low volatility (BB width < 20th percentile) in bull trend.
# Short when price breaks below lower BB after low volatility in bear trend.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Designed for 50-150 total trades over 4 years (12-37/year) with Sharpe > 0 on BTC/ETH/SOL.
# Works in bull via breakout continuation and in bear via short breakdowns with trend filter.
# Bollinger squeeze identifies low volatility periods preceding explosive moves.

name = "6h_BBandSqueeze_12hEMA50_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 6h data for Bollinger Bands (primary timeframe)
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Bollinger squeeze condition: width < 20th percentile of last 50 periods
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).quantile(0.20).values
    bb_squeeze = bb_width < bb_width_percentile
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: current 6h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(bb_squeeze[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        bb_upper_val = bb_upper[i]
        bb_lower_val = bb_lower[i]
        bb_squeeze_val = bb_squeeze[i]
        ema_trend = ema_50_12h_aligned[i]
        vol_conf = volume_confirm[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Breakout conditions
        long_breakout = close_val > bb_upper_val
        short_breakout = close_val < bb_lower_val
        
        # Entry logic
        if position == 0:
            if is_bull_trend and long_breakout and bb_squeeze_val and vol_conf:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and short_breakout and bb_squeeze_val and vol_conf:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend reversal or volatility expansion
            if close_val < ema_trend or not bb_squeeze_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend reversal or volatility expansion
            if close_val > ema_trend or not bb_squeeze_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals