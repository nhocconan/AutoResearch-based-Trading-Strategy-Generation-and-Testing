#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeRegime
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume/volatility regime.
In strong trends (price > 1d EMA50 + ATR expansion), trade breakouts in trend direction.
In weak trends/ranges (price < 1d EMA50 OR ATR contraction), fade at Donchian extremes.
Uses discrete sizing (0.25) and ATR-based stops to minimize fee churn and manage drawdown.
Targets 20-40 trades/year by requiring volume expansion and volatility regime alignment.
Works in bull via trend-following breakouts, in bear via mean reversion at extremes when volatility contracts.
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
    volume = prices['volume'].values
    
    # Get 4h data for Donchian calculations (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian channels for each 4h bar (20-period lookback)
    upper_20_4h = np.full(len(close_4h), np.nan)
    lower_20_4h = np.full(len(close_4h), np.nan)
    
    for i in range(20, len(close_4h)):
        upper_20_4h[i] = np.max(high_4h[i-20:i])  # highest high of past 20 bars
        lower_20_4h[i] = np.min(low_4h[i-20:i])   # lowest low of past 20 bars
    
    # Align Donchian levels to original timeframe
    upper_20_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_20_4h)
    lower_20_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_20_4h)
    
    # Get 1d data for trend filter and volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA50 for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d ATR(14) for volatility regime
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr_1d = np.concatenate([[np.nan], tr_1d])  # align with close_1d
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average (moderate to balance trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma_20)
    
    # Volatility regime: ATR expansion/contraction
    atr_ma_50 = pd.Series(atr_14_1d_aligned).rolling(window=50, min_periods=50).mean().values
    vol_expansion = atr_14_1d_aligned > atr_ma_50  # volatility expanding
    vol_contraction = atr_14_1d_aligned < atr_ma_50  # volatility contracting
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_20_4h_aligned[i]) or np.isnan(lower_20_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1d_aligned[i]
        atr_now = atr_14_1d_aligned[i]
        vol_exp = vol_expansion[i]
        vol_contr = vol_contraction[i]
        
        if position == 0:
            # Regime-based entry logic
            if ema_trend > 0 and vol_exp:  # Strong uptrend with expanding volatility
                # Long: break above upper Donchian with volume spike
                long_signal = (close[i] > upper_20_4h_aligned[i]) and vol_spike[i]
                # Short: break below lower Donchian only in extreme cases
                short_signal = (close[i] < lower_20_4h_aligned[i]) and vol_spike[i] and (close[i] < ema_trend * 0.97)
            elif ema_trend > 0 and vol_contr:  # Uptrend but contracting volatility (potential exhaustion)
                # Fade: short at upper Donchian, long at lower Donchian
                long_signal = (close[i] < lower_20_4h_aligned[i]) and vol_spike[i]
                short_signal = (close[i] > upper_20_4h_aligned[i]) and vol_spike[i]
            else:  # Downtrend or weak trend
                if ema_trend > 0:  # Weak uptrend
                    # Fade at extremes
                    long_signal = (close[i] < lower_20_4h_aligned[i]) and vol_spike[i]
                    short_signal = (close[i] > upper_20_4h_aligned[i]) and vol_spike[i]
                else:  # Downtrend
                    if vol_exp:  # Strong downtrend with expanding volatility
                        # Short: break below lower Donchian with volume spike
                        short_signal = (close[i] < lower_20_4h_aligned[i]) and vol_spike[i]
                        # Long: break above upper Donchian only in extreme cases
                        long_signal = (close[i] > upper_20_4h_aligned[i]) and vol_spike[i] and (close[i] > ema_trend * 1.03)
                    else:  # Downtrend with contracting volatility
                        # Fade at extremes
                        long_signal = (close[i] < lower_20_4h_aligned[i]) and vol_spike[i]
                        short_signal = (close[i] > upper_20_4h_aligned[i]) and vol_spike[i]
            
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
            # Exit conditions: touch lower Donchian or trend/volatility deterioration
            exit_signal = (close[i] < lower_20_4h_aligned[i]) or \
                         (ema_trend <= 0 and vol_contr) or \
                         (close[i] < ema_trend * 0.985)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: touch upper Donchian or trend/volatility deterioration
            exit_signal = (close[i] > upper_20_4h_aligned[i]) or \
                         (ema_trend >= 0 and vol_exp) or \
                         (close[i] > ema_trend * 1.015)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeRegime"
timeframe = "4h"
leverage = 1.0