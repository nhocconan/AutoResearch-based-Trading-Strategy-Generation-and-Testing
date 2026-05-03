#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout + 1d EMA50 trend filter + volume confirmation
# Bollinger Band squeeze identifies low volatility periods preceding explosive moves
# Breakout from squeeze with volume confirmation captures genuine momentum
# 1d EMA50 ensures we trade with higher timeframe trend to avoid whipsaws in chop
# Works in bull/bear markets: squeeze breakouts occur in all regimes, trend filter prevents counter-trend trades
# Target: 12-25 trades/year (50-100 total over 4 years) to minimize fee drag
# Bollinger Bands are effective for volatility-based breakout strategies across all market conditions

name = "6h_BollingerSqueeze_1dEMA50_VolumeBreakout"
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Bollinger Bands on 6h: 20-period, 2 standard deviations
    bb_period = 20
    bb_std = 2.0
    ma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = ma + (bb_std * std)
    lower_band = ma - (bb_std * std)
    bb_width = (upper_band - lower_band) / ma  # Normalized bandwidth
    
    # Bollinger Band squeeze: bandwidth below 20-period rolling 10th percentile
    bb_width_series = pd.Series(bb_width)
    squeeze_threshold = bb_width_series.rolling(window=20, min_periods=20).quantile(0.10).values
    is_squeeze = bb_width < squeeze_threshold
    
    # Volume confirmation: volume > 1.5 x 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ma[i]) or np.isnan(std[i]) or 
            np.isnan(squeeze_threshold[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Long signal: squeeze breakout above upper band + volume + price above 1d EMA50
        # Short signal: squeeze breakout below lower band + volume + price below 1d EMA50
        if position == 0:
            if (is_squeeze[i-1] and not is_squeeze[i] and  # Squeeze just ended
                close[i] > upper_band[i] and 
                volume_confirm[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            elif (is_squeeze[i-1] and not is_squeeze[i] and  # Squeeze just ended
                  close[i] < lower_band[i] and 
                  volume_confirm[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Bollinger Bands (mean reversion) OR squeeze reforms
            if close[i] < ma[i] or is_squeeze[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Bollinger Bands OR squeeze reforms
            if close[i] > ma[i] or is_squeeze[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals