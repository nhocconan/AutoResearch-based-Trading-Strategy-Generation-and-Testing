#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 12h Volume Spike and 1d Trend Filter
# Long when: BB Width at 20-period low (squeeze) AND price breaks above upper BB AND 12h volume > 1.5x average AND 1d close > 1d EMA50
# Short when: BB Width at 20-period low (squeeze) AND price breaks below lower BB AND 12h volume > 1.5x average AND 1d close < 1d EMA50
# Exit when price returns to middle BB (mean reversion in squeeze)
# Bollinger Squeeze identifies low volatility primed for expansion
# Volume spike confirms institutional participation
# 1d EMA50 filter ensures alignment with higher timeframe trend
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25 to minimize fee churn

name = "6h_BollingerSqueeze_VolumeSpike_1dTrend"
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
    
    # Get 12h data ONCE before loop for volume average
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:  # Need enough for volume average
        return np.zeros(n)
    volume_12h = df_12h['volume'].values
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 12h average volume (20-period)
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Bollinger Bands (20, 2) on 6h
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    
    # BB Width percentile lookback (50 periods) to identify squeeze
    # Squeeze when BB Width is at 20-period low (bottom 20% of last 50 periods)
    bb_width_series = pd.Series(bb_width)
    bb_width_rank = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 50 else np.nan, raw=False
    ).values
    squeeze_condition = bb_width_rank <= 0.2  # Bottom 20% = squeeze
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(vol_ma_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(sma_20[i]) or np.isnan(std_20[i]) or 
            np.isnan(squeeze_condition[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Squeeze + break above upper BB + volume spike + 1d uptrend
            if (squeeze_condition[i] and 
                close[i] > upper_bb[i] and 
                volume[i] > 1.5 * vol_ma_aligned[i] and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Squeeze + break below lower BB + volume spike + 1d downtrend
            elif (squeeze_condition[i] and 
                  close[i] < lower_bb[i] and 
                  volume[i] > 1.5 * vol_ma_aligned[i] and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: return to middle BB (mean reversion) or stoploss
            if close[i] < sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to middle BB (mean reversion) or stoploss
            if close[i] > sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals