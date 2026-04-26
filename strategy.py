#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_RegimeFilter
Hypothesis: On 4h timeframe, use Camarilla R1/S1 from 1d pivot points for breakout entries with 1d EMA34 trend filter and volume confirmation (>1.5x 20-period average). Add choppiness regime filter (CHOP > 50 for mean reversion, CHOP <= 50 for trend following) to adapt to bull/bear/choppy markets. Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 periods for EMA34
        return np.zeros(n)
    
    # Calculate 1d OHLC for Camarilla pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (based on previous 1d bar's range)
    # Camarilla R1 = close + 1.1*(high - low)/12
    # Camarilla S1 = close - 1.1*(high - low)/12
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    
    # Set first value to NaN (no previous bar)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_r1 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 12
    camarilla_s1 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 12
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d ATR14 for choppiness indicator
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(np.maximum(tr1, tr2), tr3)])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all 1d indicators to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate choppiness index on 1d: CHOP = 100 * log10(sum(ATR14) / (max(high)-min(low)) over period) / log10(period)
    # Using 14-period CHOP: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    high_14_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    low_14_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    sum_atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop_denom = np.maximum(high_14_1d - low_14_1d, 1e-10)
    chop_ratio = np.divide(sum_atr_14_1d, chop_denom, out=np.zeros_like(sum_atr_14_1d), where=chop_denom!=0)
    chop_ratio = np.maximum(chop_ratio, 1e-10)
    chop_14_1d = 100 * np.log10(chop_ratio) / np.log10(14)
    chop_14_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_14_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need all indicators ready
    start_idx = max(34, 20, 14)  # 34 for EMA34, 20 for volume MA, 14 for ATR/CHOP
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_14_1d_aligned[i]) or
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        chop = chop_14_1d_aligned[i]
        trend_1d_uptrend = close[i] > ema_34_1d_aligned[i]
        trend_1d_downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Regime-adaptive entry logic
            if chop > 50:  # Ranging market: mean reversion at Camarilla levels
                # Long: price drops to S1 then reverses up (bullish engulfing proxy)
                long_signal = (close[i] < camarilla_s1_aligned[i] and 
                              close[i] > open_prices[i] if 'open' in prices.columns else True) and \
                             volume_spike[i]
                # Short: price rises to R1 then reverses down (bearish engulfing proxy)
                short_signal = (close[i] > camarilla_r1_aligned[i] and 
                               close[i] < open_prices[i] if 'open' in prices.columns else True) and \
                              volume_spike[i]
            else:  # Trending market: breakout in direction of trend
                # Long: price breaks above R1 with uptrend
                long_signal = (close[i] > camarilla_r1_aligned[i]) and trend_1d_uptrend and volume_spike[i]
                # Short: price breaks below S1 with downtrend
                short_signal = (close[i] < camarilla_s1_aligned[i]) and trend_1d_downtrend and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit conditions: adaptive to regime
            if chop > 50:  # Ranging: mean reversion exit
                if close[i] > camarilla_r1_aligned[i]:  # Reached opposite level
                    signals[i] = 0.0
                    position = 0
            else:  # Trending: trend-following exit
                if not trend_1d_uptrend or close[i] < camarilla_s1_aligned[i]:  # Trend broken or reached stop
                    signals[i] = 0.0
                    position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit conditions: adaptive to regime
            if chop > 50:  # Ranging: mean reversion exit
                if close[i] < camarilla_s1_aligned[i]:  # Reached opposite level
                    signals[i] = 0.0
                    position = 0
            else:  # Trending: trend-following exit
                if not trend_1d_downtrend or close[i] > camarilla_r1_aligned[i]:  # Trend broken or reached stop
                    signals[i] = 0.0
                    position = 0
    
    return signals

# Extract open prices for engulfing pattern check
open_prices = prices['open'].values if 'open' in prices.columns else close

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0