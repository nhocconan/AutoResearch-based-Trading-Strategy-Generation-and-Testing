#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above Donchian upper band AND 1d EMA34 uptrend AND volume > 2.0x 20-bar average
# Short when price breaks below Donchian lower band AND 1d EMA34 downtrend AND volume > 2.0x 20-bar average
# Exit when price crosses the Donchian middle band (20-period midpoint)
# Uses discrete position sizing (0.25) to minimize fee churn. Works in both bull/bear by following 1d trend.
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag.

name = "4h_Donchian20_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian Channel (20) on 4h data
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_band = highest_high_20
    lower_band = lowest_low_20
    middle_band = (upper_band + lower_band) / 2  # 20-period midpoint
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # warmup for Donchian and EMA34
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema34_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_upper = upper_band[i]
        curr_lower = lower_band[i]
        curr_middle = middle_band[i]
        curr_ema34 = ema34_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend regime: bullish if price > 1d EMA34, bearish if price < 1d EMA34
        is_bullish_regime = curr_close > curr_ema34
        is_bearish_regime = curr_close < curr_ema34
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: price > upper band AND bullish regime
                if curr_close > curr_upper and is_bullish_regime:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price < lower band AND bearish regime
                elif curr_close < curr_lower and is_bearish_regime:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit when price crosses below middle band
            if curr_close < curr_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit when price crosses above middle band
            if curr_close > curr_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals