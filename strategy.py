#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout + 12h EMA50 Trend Filter + Volume Spike
# Long when: BB width at 20-period low + price breaks above upper BB + price > 12h EMA50 + volume > 2.0x avg
# Short when: BB width at 20-period low + price breaks below lower BB + price < 12h EMA50 + volume > 2.0x avg
# Uses discrete sizing (0.25) to minimize fee churn. Works in bull/bear via 12h trend filter.
# Timeframe: 6h (primary), HTF: 12h for EMA50 trend.

name = "6h_BBSqueeze_12hEMA50_VolumeSpike_v1"
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
    
    # Load HTF data ONCE before loop for 12h EMA50 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Bollinger Bands (20, 2.0)
    close_s = pd.Series(close)
    bb_mid = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_width = bb_upper - bb_lower
    
    # BB width percentile (20-period lookback for squeeze)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # warmup for BB and EMA50
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_50_12h_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_bb_width_percentile = bb_width_percentile[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Price falls below middle BB (mean reversion)
            # 2. Price falls below 12h EMA50 (trend change)
            if (curr_close < bb_mid[i] or
                curr_close < curr_ema_50_12h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Price rises above middle BB (mean reversion)
            # 2. Price rises above 12h EMA50 (trend change)
            if (curr_close > bb_mid[i] or
                curr_close > curr_ema_50_12h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: BB squeeze (width < 20th percentile) AND price breaks above upper BB AND price > 12h EMA50 AND volume confirm
            if (curr_bb_width_percentile < 20.0 and
                curr_high > bb_upper[i] and
                curr_close > curr_ema_50_12h and
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: BB squeeze (width < 20th percentile) AND price breaks below lower BB AND price < 12h EMA50 AND volume confirm
            elif (curr_bb_width_percentile < 20.0 and
                  curr_low < bb_lower[i] and
                  curr_close < curr_ema_50_12h and
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals