#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band Squeeze with Volume Spike and 1d Trend Filter
# Bollinger Band width (BBW) contraction indicates low volatility, often preceding breakout.
# A volume spike confirms institutional interest in the breakout direction.
# 1d EMA (50) filters trades to align with the higher timeframe trend.
# This combination captures explosive moves after consolidation periods, effective in both bull and bear markets.
# Target: 25-35 trades/year (100-140 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA (50) for trend direction
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Bollinger Bands (20, 2) on 12h
    bb_period = 20
    bb_std = 2.0
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + bb_std * std
    lower_band = sma - bb_std * std
    bb_width = (upper_band - lower_band) / sma  # Normalized bandwidth
    
    # Bollinger Band width percentile (50-period) to detect squeeze
    bbw_series = pd.Series(bb_width)
    bbw_percentile = bbw_series.rolling(window=50, min_periods=50).rank(pct=True).values
    
    # Volume spike detection (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma  # Current volume relative to average
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for BBW percentile and other indicators
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(sma[i]) or np.isnan(bbw_percentile[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: only trade in direction of 1d EMA
        above_ema = price > ema_1d_aligned[i]
        
        if position == 0:
            # Look for Bollinger Band squeeze (low volatility) followed by volume spike and breakout
            squeeze_condition = bbw_percentile[i] <= 0.2  # Bottom 20% of BBW = squeeze
            volume_spike = vol_ratio[i] >= 2.0  # Volume at least 2x average
            
            if squeeze_condition and volume_spike:
                # Breakout above upper band with uptrend filter -> long
                if price > upper_band[i] and above_ema:
                    position = 1
                    signals[i] = position_size
                # Breakout below lower band with downtrend filter -> short
                elif price < lower_band[i] and not above_ema:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle band (mean reversion) or trend changes
            if price <= sma[i] or price < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to middle band (mean reversion) or trend changes
            if price >= sma[i] or price > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Bollinger_Squeeze_Volume_Spike_1dEMA"
timeframe = "12h"
leverage = 1.0