#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with Daily Regime Filter
# Bollinger Band squeeze (low volatility) precedes expansion breakouts in both bull and bear markets
# Daily trend filter (price vs 50 EMA) ensures we trade breakouts in direction of higher timeframe trend
# Volume confirmation validates breakout strength
# Works in all regimes: squeeze breakouts capture volatility expansion after consolidation
# Target: 12-25 trades/year (48-100 total over 4 years)

name = "6h_BBSqueeze_DailyTrend_VolumeConfirm_v1"
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
    
    # Load HTF data ONCE before loop for daily calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Bollinger Bands (20, 2.0) on 6h data
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + (2.0 * bb_std)
    bb_lower = bb_middle - (2.0 * bb_std)
    
    # Bollinger Band Width: (Upper - Lower) / Middle
    bb_width = np.where(bb_middle != 0, (bb_upper - bb_lower) / bb_middle, 0)
    
    # Bollinger Band Squeeze: BB Width < 5th percentile of last 50 periods
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).quantile(0.05).values
    squeeze_condition = bb_width < bb_width_percentile
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 50, 20, 20)  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(bb_width[i]) or np.isnan(bb_width_percentile[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_bb_upper = bb_upper[i]
        curr_bb_lower = bb_lower[i]
        curr_squeeze = squeeze_condition[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema50_1d = ema50_1d_aligned[i]
        
        # Determine trend regime from daily EMA50
        bullish_regime = curr_close > curr_ema50_1d
        bearish_regime = curr_close < curr_ema50_1d
        
        if position == 0:  # Flat - look for new entries
            # Look for breakout after Bollinger Band squeeze
            if not curr_squeeze and curr_volume_confirm:
                # Bullish breakout: price breaks above upper BB in bullish regime
                if bullish_regime and curr_close > curr_bb_upper:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below lower BB in bearish regime
                elif bearish_regime and curr_close < curr_bb_lower:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price crosses below middle BB OR squeeze returns (low volatility)
            if curr_close < bb_middle[i] or squeeze_condition[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price crosses above middle BB OR squeeze returns (low volatility)
            if curr_close > bb_middle[i] or squeeze_condition[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals