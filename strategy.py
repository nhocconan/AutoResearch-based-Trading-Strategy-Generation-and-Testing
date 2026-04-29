#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Donchian(20) provides clear structure-based breakout levels from prior 20 periods (~5 days)
# Breakout above upper band or below lower band indicates strong momentum with high probability continuation
# 1d EMA34 provides intermediate trend filter to avoid counter-trend trades
# Volume spike confirms institutional participation
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Works in both bull and bear markets by following 1d trend while capturing 6h momentum

name = "6h_Donchian20_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h timeframe (completed 1d bar only)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian(20) channels from 6h data (prior 20 periods)
    # Upper band = highest high of prior 20 periods
    # Lower band = lowest low of prior 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: volume > 2.0x 20-period average (20*6h = ~5 days)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for EMA34 and Donchian
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema34_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema34 = ema34_aligned[i]
        curr_dc_upper = donchian_upper[i]
        curr_dc_lower = donchian_lower[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend regime: bullish if price > 1d EMA34, bearish if price < 1d EMA34
        is_bullish_regime = curr_close > curr_ema34
        is_bearish_regime = curr_close < curr_ema34
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: break above Donchian upper band AND bullish regime
                if curr_high > curr_dc_upper and is_bullish_regime:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: break below Donchian lower band AND bearish regime
                elif curr_low < curr_dc_lower and is_bearish_regime:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price breaks below Donchian lower band (reversal) OR regime changes to bearish
            if curr_low < curr_dc_lower or not is_bullish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price breaks above Donchian upper band (reversal) OR regime changes to bullish
            if curr_high > curr_dc_upper or not is_bearish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals