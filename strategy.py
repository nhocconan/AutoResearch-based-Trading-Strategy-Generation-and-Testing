#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 12h trend filter and volume confirmation
# Bollinger Squeeze (BBWidth < 20th percentile) identifies low volatility primed for breakout
# Breakout direction determined by 12h EMA50 trend: long in uptrend, short in downtrend
# Volume spike (>2x 20-period EMA) confirms institutional participation
# Designed for low trade frequency: ~10-20 trades/year per symbol with 0.25 sizing
# Works in bull/bear markets by only trading breakouts in the direction of the 12h trend

name = "6h_BollingerSqueeze_Breakout_12hEMA50_Trend_Volume_v1"
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
    
    # 12h HTF data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Bollinger Bands (20, 2) on 6h
    close_s = pd.Series(close)
    sma_20 = close_s.rolling(window=20, min_periods=20).mean().values
    std_20 = close_s.rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized BB width
    
    # BB Squeeze: BBWidth < 20th percentile (lookback 50 periods)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).quantile(0.20).values
    bb_squeeze = bb_width < bb_width_percentile
    
    # Volume confirmation: volume > 2.0 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(60, 50, 20, 20)  # Need 12h EMA50, BB (20), volume EMA (20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(sma_20[i]) or 
            np.isnan(std_20[i]) or np.isnan(bb_width_percentile[i]) or 
            np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade in direction of 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if uptrend and bb_squeeze[i]:
                # Long: price breaks above upper BB with volume spike
                if close[i] > upper_bb[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif downtrend and bb_squeeze[i]:
                # Short: price breaks below lower BB with volume spike
                if close[i] < lower_bb[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # No squeeze or wrong direction
        
        elif position == 1:  # Long position
            # Exit: price closes below middle BB (mean reversion) or loss of momentum
            if close[i] < sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above middle BB (mean reversion) or loss of momentum
            if close[i] > sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals