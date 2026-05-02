#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d ADX trend filter and volume confirmation
# Uses 12h primary timeframe targeting 12-37 trades/year (50-150 total over 4 years)
# 1d ADX > 25 provides strong trend filter to avoid whipsaws in ranging markets
# 12h Donchian(20) breakout captures momentum with clear structure
# Volume confirmation (>1.3 * 20-period EMA) ensures institutional participation
# Discrete position sizing (0.25) minimizes fee churn while maintaining exposure
# Works in bull (continuation breakouts) and bear (trend continuation shorts) markets
# Designed to avoid overtrading by requiring confluence of price structure, trend strength, and volume

name = "12h_Donchian20_1dADX25_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for Donchian calculation and volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for ADX
        return np.zeros(n)
    
    # 12h Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 1d ADX calculation (14-period)
    plus_dm = np.where((df_1d['high'].diff()) > (df_1d['low'].diff().abs()), 
                       np.maximum(df_1d['high'].diff(), 0), 0)
    minus_dm = np.where((df_1d['low'].diff().abs()) > (df_1d['high'].diff()), 
                        np.maximum(-df_1d['low'].diff(), 0), 0)
    
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_14 = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    plus_di_14 = 100 * (plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_14)
    minus_di_14 = 100 * (minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_14)
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_14 = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_14_values = adx_14.values
    
    # Align 1d ADX to 12h timeframe
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14_values)
    
    # Volume confirmation: volume > 1.3 * 20-period EMA (12h)
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for all indicators
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d ADX (strong trend when ADX > 25)
        strong_trend = adx_14_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            if strong_trend:
                # Long: price breaks above Donchian high with volume spike
                if close[i] > donchian_high[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Donchian low with volume spike
                elif close[i] < donchian_low[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # No entries in weak trend/ranging markets
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian low or ADX falls below 20 (trend weakening)
            if close[i] < donchian_low[i] or adx_14_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or ADX falls below 20 (trend weakening)
            if close[i] > donchian_high[i] or adx_14_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals