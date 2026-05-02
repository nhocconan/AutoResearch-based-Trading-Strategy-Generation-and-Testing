#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter
# Uses Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) to measure bull/bear strength
# 1d ADX > 25 filters for trending markets, < 20 for ranging markets
# In trending markets (ADX > 25): take Elder Ray signals in trend direction
# In ranging markets (ADX < 20): fade Elder Ray extremes (mean reversion)
# Discrete position sizing 0.25 balances risk and minimizes fee churn
# Targets 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by adapting to regime

name = "6h_ElderRay_1dADX_Regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for regime detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close_1d index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM-
    tr_period = 14
    tr_smooth = pd.Series(tr).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Align ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13  # negative values indicate bear strength
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for EMA and ADX)
    start_idx = 30  # max(13 for EMA, 30 for ADX) 
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if adx_val > 25:  # Trending regime
                # Long: Bull Power rising and positive (bulls in control)
                if bull_power[i] > 0 and bull_power[i] > bull_power[i-1]:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power falling and negative (bears in control)
                elif bear_power[i] < 0 and bear_power[i] < bear_power[i-1]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif adx_val < 20:  # Ranging regime
                # Long: Bull Power extremely negative (oversold, mean reversion long)
                if bull_power[i] < 0 and bull_power[i] < np.percentile(bull_power[max(0,i-50):i+1], 10):
                    signals[i] = 0.25
                    position = 1
                # Short: Bull Power extremely positive (overbought, mean reversion short)
                elif bull_power[i] > 0 and bull_power[i] > np.percentile(bull_power[max(0,i-50):i+1], 90):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:  # Transition regime (20 <= ADX <= 25) - no trades
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions
            if adx_val > 25:  # Trending: exit when bull power weakens
                if bull_power[i] < 0 or bull_power[i] < bull_power[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif adx_val < 20:  # Ranging: exit when mean reversion complete
                if bull_power[i] > np.percentile(bull_power[max(0,i-20):i+1], 50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Transition: hold until clear signal
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            if adx_val > 25:  # Trending: exit when bear power weakens
                if bear_power[i] > 0 or bear_power[i] > bear_power[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            elif adx_val < 20:  # Ranging: exit when mean reversion complete
                if bull_power[i] < np.percentile(bull_power[max(0,i-20):i+1], 50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Transition: hold until clear signal
                signals[i] = -0.25
    
    return signals