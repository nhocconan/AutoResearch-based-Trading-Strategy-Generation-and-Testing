#!/usr/bin/env python3
"""
6h_ADX_DMI_Trend_12hVolRegime_v1
Hypothesis: Trade 6h ADX/DMI trend strength with 12h volume regime filter.
ADX > 25 indicates strong trend, DMI crossover gives direction. 12h volume regime (above/below median) acts as trend filter:
- In high volume regime: only take trend-following signals
- In low volume regime: only take mean-reversion at extremes
This adapts to both trending (bull/bear) and ranging markets, reducing whipsaw.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HTF volume regime
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h volume median for regime detection
    vol_12h = df_12h['volume'].values
    vol_median_12h = pd.Series(vol_12h).rolling(window=50, min_periods=50).median().values
    vol_median_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_median_12h)
    
    # Calculate ADX(14) and DMI(14) on 6h
    period = 14
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    # Smoothed TR, +DM, -DM
    def ma(arr, n):
        return pd.Series(arr).ewm(alpha=1/n, adjust=False, min_periods=n).mean().values
    tr_ma = ma(tr, period)
    plus_dm_ma = ma(plus_dm, period)
    minus_dm_ma = ma(minus_dm, period)
    # +DI, -DI, DX, ADX
    plus_di = 100 * plus_dm_ma / tr_ma
    minus_di = 100 * minus_dm_ma / tr_ma
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = ma(dx, period)
    
    # Align indicators (already computed on 6h, no HTF alignment needed for same TF)
    # But we need to handle warmup periods
    # Volume regime: 1 = high volume (above median), 0 = low volume
    vol_regime = (volume > vol_median_12h_aligned).astype(float)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for ADX/DMI (2*period for stability)
    start_idx = 2 * period
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or
            np.isnan(vol_median_12h_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine trend direction and strength
        bullish_cross = plus_di[i] > minus_di[i]
        bearish_cross = minus_di[i] > plus_di[i]
        strong_trend = adx[i] > 25
        
        if position == 0:
            # Entry logic depends on volume regime
            if vol_regime[i] > 0.5:  # High volume regime: trend following
                if bullish_cross and strong_trend:
                    signals[i] = 0.25
                    position = 1
                elif bearish_cross and strong_trend:
                    signals[i] = -0.25
                    position = -1
            else:  # Low volume regime: mean reversion at extremes
                # Use price relative to recent high/low for mean reversion
                lookback = 20
                if i >= lookback:
                    recent_high = np.max(high[i-lookback:i+1])
                    recent_low = np.min(low[i-lookback:i+1])
                    price_range = recent_high - recent_low
                    if price_range > 0:
                        price_pos = (close[i] - recent_low) / price_range
                        # Extreme oversold in low vol -> long
                        if price_pos < 0.2 and bullish_cross:
                            signals[i] = 0.25
                            position = 1
                        # Extreme overbought in low vol -> short
                        elif price_pos > 0.8 and bearish_cross:
                            signals[i] = -0.25
                            position = -1
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: trend weakens OR reverse crossover
            if not strong_trend or bearish_cross:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: trend weakens OR reverse crossover
            if not strong_trend or bullish_cross:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_DMI_Trend_12hVolRegime_v1"
timeframe = "6h"
leverage = 1.0