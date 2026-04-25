#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_Regime
Hypothesis: On 6h timeframe, Elder Ray Bull/Bear Power combined with 1d EMA trend filter and Bollinger Bandwidth regime filter captures momentum in trending markets while avoiding whipsaws in ranging markets. Bull Power > 0 + Bear Power < 0 indicates strong momentum aligned with 1d trend. Low volatility regime (BBW < 30th percentile) avoids false signals. Discrete sizing (0.25) minimizes fee churn. Works in bull markets via long signals aligned with uptrend and bear markets via short signals aligned with downtrend. Uses 1d HTF for trend and regime to avoid look-ahead.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for HTF trend and regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on 1d for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Bollinger Bands on 1d for regime filter (BBW = (upper-lower)/middle)
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + 2.0 * std_20_1d
    lower_bb_1d = sma_20_1d - 2.0 * std_20_1d
    bb_width_1d = (upper_bb_1d - lower_bb_1d) / sma_20_1d
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width_1d)
    
    # Calculate 1d percentile rank of BBW (lookback 50 periods)
    bb_width_percentile = np.full_like(bb_width_aligned, np.nan)
    for i in range(50, len(bb_width_aligned)):
        window = bb_width_aligned[i-50:i]
        if not np.all(np.isnan(window)):
            percentile = np.nanpercentile(window, bb_width_aligned[i])
            bb_width_percentile[i] = percentile
    
    # Calculate Elder Ray on 6h: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = max(50, 34, 20, 13)  # BBW percentile, EMA34, BBands, EMA13
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(bb_width_percentile[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_val = ema_34_aligned[i]
        bb_percentile = bb_width_percentile[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        close_val = close[i]
        
        # Regime condition: low volatility (BBW < 30th percentile) for better signal quality
        low_vol_regime = bb_percentile < 30.0
        
        if position == 0:
            # Look for entry signals: Elder Ray alignment with 1d trend and low volatility regime
            # Long: Bull Power > 0 (strong buying pressure) + Bear Power < 0 (weak selling pressure) + 
            #       price above 1d EMA (uptrend) + low volatility regime
            long_signal = (bull_val > 0) and (bear_val < 0) and (close_val > ema_val) and low_vol_regime
            # Short: Bull Power < 0 (weak buying pressure) + Bear Power > 0 (strong selling pressure) + 
            #        price below 1d EMA (downtrend) + low volatility regime
            short_signal = (bull_val < 0) and (bear_val > 0) and (close_val < ema_val) and low_vol_regime
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Loss of momentum: Bull Power <= 0 or Bear Power >= 0 (momentum divergence)
            if bull_val <= 0 or bear_val >= 0:
                signals[i] = 0.0
                position = 0
            # 2. Trend reversal: price crosses below 1d EMA
            elif close_val < ema_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Loss of momentum: Bull Power >= 0 or Bear Power <= 0 (momentum divergence)
            if bull_val >= 0 or bear_val <= 0:
                signals[i] = 0.0
                position = 0
            # 2. Trend reversal: price crosses above 1d EMA
            elif close_val > ema_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_Regime"
timeframe = "6h"
leverage = 1.0