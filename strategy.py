#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly Bollinger Band squeeze breakout with volume confirmation and 1-week trend filter.
# Uses Bollinger Band width percentile to detect low volatility (squeeze) and breaks out in direction of weekly trend.
# Volume spike confirms breakout authenticity. Designed for low-frequency, high-conviction trades.
# Works in bull/bear markets by requiring trend alignment and volatility contraction before breakout.
# Target: 10-25 trades/year to minimize fee drag.

name = "1d_BollingerSqueeze_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend: EMA21
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Bollinger Bands (20, 2)
    bb_window = 20
    bb_std = 2
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=bb_window, min_periods=bb_window).mean().values
    bb_std_dev = close_s.rolling(window=bb_window, min_periods=bb_window).std().values
    bb_upper = bb_middle + bb_std * bb_std_dev
    bb_lower = bb_middle - bb_std * bb_std_dev
    
    # Bollinger Band Width and its percentile (for squeeze detection)
    bb_width = bb_upper - bb_lower
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(bb_width_percentile[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bollinger Band squeeze: width below 20th percentile (low volatility)
            squeeze_condition = bb_width_percentile[i] < 20
            
            # Long: break above upper band + weekly uptrend (price > weekly EMA21) + volume spike
            long_cond = squeeze_condition and \
                        (close[i] > bb_upper[i]) and \
                        (close[i] > ema_21_1w_aligned[i]) and \
                        volume_spike[i]
            # Short: break below lower band + weekly downtrend (price < weekly EMA21) + volume spike
            short_cond = squeeze_condition and \
                         (close[i] < bb_lower[i]) and \
                         (close[i] < ema_21_1w_aligned[i]) and \
                         volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below middle band (mean reversion)
            if close[i] < bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above middle band (mean reversion)
            if close[i] > bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals