#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power (Bull/Bear) with 1d ADX25 regime filter and 13-period EMA trend
# Uses Elder Ray to measure bull/bear power relative to 13-period EMA, ADX(25) from 1d to filter trending vs ranging markets,
# and EMA(13) on 6h for dynamic trend direction. In trending markets (ADX>25), take signals in direction of EMA(13).
# In ranging markets (ADX<25), fade extremes when Bull/Bear power diverges from price.
# Designed to work in both bull and bear regimes by adapting to market conditions.
# Target: 12-37 trades/year via regime-adaptive filtering

name = "6h_ElderRay_Power_1dADX25_Regime_EMA13Trend_v1"
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
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for ADX calculation
        return np.zeros(n)
    
    # Calculate ADX(25) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d[1:] - low_1d[:-1])
    tr2 = pd.Series(np.abs(high_1d[1:] - close_1d[:-1]))
    tr3 = pd.Series(np.abs(low_1d[1:] - close_1d[:-1]))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    tr_1d = np.concatenate([[np.nan], tr_1d])  # Align with 1d index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_period = 25
    tr_smooth = pd.Series(tr_1d).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Align 1d ADX to 6h timeframe (completed 1d candles only)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 13-period EMA on 6h close
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray Power (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 25)  # Need sufficient history for EMA and ADX
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        ema_val = ema_13[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        price = close[i]
        
        # Regime determination: ADX > 25 = trending, ADX < 25 = ranging
        is_trending = adx_val > 25
        
        if is_trending:
            # Trending regime: follow EMA13 direction
            if price > ema_val and bull_val > 0:  # Uptrend confirmation
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            elif price < ema_val and bear_val < 0:  # Downtrend confirmation
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            else:
                # Exit if trend weakens
                if position == 1 and (price <= ema_val or bull_val <= 0):
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and (price >= ema_val or bear_val >= 0):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25 if position == -1 else 0.0
        else:
            # Ranging regime: mean reversion at extremes
            # Look for divergences: price making new high/low but power not confirming
            if position == 0:
                # Long setup: price makes new low but bear power doesn't confirm (bullish divergence)
                if i >= 20:
                    recent_low = np.min(low[i-20:i+1])
                    recent_bear = np.min(bear_power[i-20:i+1])
                    if price == recent_low and bear_val > recent_bear:  # Bullish divergence
                        signals[i] = 0.25
                        position = 1
                # Short setup: price makes new high but bull power doesn't confirm (bearish divergence)
                elif i >= 20:
                    recent_high = np.max(high[i-20:i+1])
                    recent_bull = np.max(bull_power[i-20:i+1])
                    if price == recent_high and bull_val < recent_bull:  # Bearish divergence
                        signals[i] = -0.25
                        position = -1
            elif position == 1:
                # Exit long when price reaches EMA13 or bear power turns positive
                if price >= ema_val or bear_val >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short when price reaches EMA13 or bull power turns negative
                if price <= ema_val or bull_val <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals