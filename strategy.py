#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dTrend_ATRFilter_v1
Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and ATR volatility expansion filter.
- Long when price breaks above Camarilla R3 AND 1d EMA34 uptrend AND ATR(14) > ATR(50)
- Short when price breaks below Camarilla S3 AND 1d EMA34 downtrend AND ATR(14) > ATR(50)
- Uses Camarilla levels from prior completed 6h bar for structure-based breakouts
- 1d EMA34 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws
- ATR(14) > ATR(50) confirms volatility expansion (institutional participation)
- Designed for moderate frequency (target 12-37 trades/year) to minimize fee drag
- Exit on opposite Camarilla level (R3/S3) touch or trend reversal
- Novelty: Combines Camarilla breakouts with HTF trend and volatility expansion filter for BTC/ETH edge in both bull/bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 6h data ONCE before loop for Camarilla levels (structure)
    df_6h = get_htf_data(prices, '6h')
    
    # Calculate prior 6h bar's OHLC for Camarilla levels
    # Camarilla levels based on prior bar's range
    lookback = 1  # Use immediately prior completed 6h bar
    prior_high = pd.Series(df_6h['high'].values).shift(1).values  # Prior bar high
    prior_low = pd.Series(df_6h['low'].values).shift(1).values    # Prior bar low
    prior_close = pd.Series(df_6h['close'].values).shift(1).values  # Prior bar close
    
    # Calculate Camarilla levels for prior 6h bar
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    prior_range = prior_high - prior_low
    camarilla_r3 = prior_close + (prior_range * 1.1 / 4)
    camarilla_s3 = prior_close - (prior_range * 1.1 / 4)
    
    # Align Camarilla levels to 6h timeframe (no additional delay needed for structure)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_s3)
    
    # Load daily data ONCE before loop for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA34 for trend filter (needs completed daily candle)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    # Trend: 1 = uptrend (close > EMA34), -1 = downtrend (close < EMA34), 0 = neutral/invalid
    trend_1d = np.where(ema_34_1d_aligned > 0, 
                        np.where(close > ema_34_1d_aligned, 1, -1), 
                        0)
    
    # Calculate ATR filter: ATR(14) > ATR(50) for volatility expansion
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14) and ATR(50)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_expansion = atr_14 > atr_50  # Volatility expansion filter
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for ATR, 34 for EMA, 1 for prior bar)
    start_idx = max(50, 34, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(trend_1d[i]) or np.isnan(atr_expansion[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla R3/S3 breakout conditions with trend and volatility expansion filter
        if position == 0:
            # Long: Price breaks above Camarilla R3 AND daily uptrend AND vol expansion
            if close[i] > camarilla_r3_aligned[i] and trend_1d[i] == 1 and atr_expansion[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3 AND daily downtrend AND vol expansion
            elif close[i] < camarilla_s3_aligned[i] and trend_1d[i] == -1 and atr_expansion[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Camarilla S3 OR daily trend turns down
            if close[i] < camarilla_s3_aligned[i] or trend_1d[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Camarilla R3 OR daily trend turns up
            if close[i] > camarilla_r3_aligned[i] or trend_1d[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_ATRFilter_v1"
timeframe = "6h"
leverage = 1.0