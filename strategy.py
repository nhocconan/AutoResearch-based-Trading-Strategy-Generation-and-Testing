#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Bollinger Band squeeze breakout with 4h EMA50 trend filter and volume spike confirmation
# Uses 1h primary timeframe targeting 15-37 trades/year (60-150 total over 4 years)
# Bollinger Band squeeze (low volatility) precedes breakouts in both bull and bear markets
# 4h EMA50 provides trend filter to avoid counter-trend entries
# Volume spike (>2.0 * 20-period EMA on 1h) confirms strong participation
# Discrete position sizing (0.20) minimizes fee churn while maintaining adequate exposure
# Works in bull (continuation via trend filter) and bear (mean reversion via squeeze breakout)
# Designed to avoid overtrading by requiring confluence of squeeze, trend, and volume

name = "1h_BollingerSqueeze_Breakout_4hEMA50_Trend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Bollinger Bands (20, 2.0) on 1h
    bb_period = 20
    bb_std = 2.0
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = close_s.rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_middle + (bb_std * bb_std_dev)
    bb_lower = bb_middle - (bb_std * bb_std_dev)
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band squeeze: width < 20-period EMA of width (low volatility regime)
    bb_width_series = pd.Series(bb_width)
    bb_width_ema = bb_width_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    bb_squeeze = bb_width < bb_width_ema
    
    # Volume confirmation: volume > 2.0 * 20-period EMA (1h)
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for all indicators
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(bb_middle[i]) or np.isnan(bb_width[i]) or np.isnan(bb_width_ema[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 4h EMA50
        bullish_bias = close[i] > ema_50_4h_aligned[i]
        bearish_bias = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bb_squeeze[i]:
                # Breakout long: price above upper BB with volume spike and bullish bias
                if close[i] > bb_upper[i] and volume_spike[i] and bullish_bias:
                    signals[i] = 0.20
                    position = 1
                # Breakout short: price below lower BB with volume spike and bearish bias
                elif close[i] < bb_lower[i] and volume_spike[i] and bearish_bias:
                    signals[i] = -0.20
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # No squeeze, wait for low volatility condition
        
        elif position == 1:  # Long position
            # Exit: price below middle BB or bearish bias on 4h
            if close[i] < bb_middle[i] or not bullish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price above middle BB or bullish bias on 4h
            if close[i] > bb_middle[i] or not bearish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals