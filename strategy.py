#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter with Donchian(20) breakout and volume confirmation.
# In trending markets (CHOP < 38.2): follow breakouts (long on upper band break, short on lower band break).
# In ranging markets (CHOP > 61.8): mean-revert at Donchian bands (short near upper band, long near lower band).
# Uses volume > 1.3x 20-period average for confirmation. Avoids whipsaws in strong trends and chop.
# Target: 20-50 trades/year by requiring regime alignment + breakout/reversion + volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate Choppiness Index (14-period) - higher = more ranging
    atr_list = []
    for i in range(n):
        if i < 1:
            atr_list.append(0)
        else:
            tr = max(
                prices['high'].iloc[i] - prices['low'].iloc[i],
                abs(prices['high'].iloc[i] - prices['close'].iloc[i-1]),
                abs(prices['low'].iloc[i] - prices['close'].iloc[i-1])
            )
            atr_list.append(tr)
    
    atr_series = pd.Series(atr_list)
    atr_sum = atr_series.rolling(window=14, min_periods=14).sum()
    highest_high = prices['high'].rolling(window=14, min_periods=14).max()
    lowest_low = prices['low'].rolling(window=14, min_periods=14).min()
    range_max_min = highest_high - lowest_low
    
    # Avoid division by zero
    chop_raw = 100 * np.log10(atr_sum / range_max_min) / np.log10(14)
    chop = chop_raw.replace([np.inf, -np.inf], np.nan).fillna(50).values  # default to middle when undefined
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(chop[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian channels (20-period)
        lookback_start = max(0, i - 19)
        high_window = prices['high'].iloc[lookback_start:i+1].values
        low_window = prices['low'].iloc[lookback_start:i+1].values
        donchian_high = np.max(high_window)
        donchian_low = np.min(low_window)
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume > 1.3 * vol_ma[i]
        
        # Regime classification
        is_trending = chop[i] < 38.2
        is_ranging = chop[i] > 61.8
        # Neutral zone (38.2-61.8): no trades to avoid whipsaw
        
        if position == 0:
            if is_trending and volume_confirm:
                # Trending market: follow breakouts
                if price > donchian_high:
                    signals[i] = 0.25
                    position = 1
                elif price < donchian_low:
                    signals[i] = -0.25
                    position = -1
            elif is_ranging and volume_confirm:
                # Ranging market: mean revert at extremes
                # Short near upper band, long near lower band
                if price >= donchian_high * 0.995:  # within 0.5% of upper band
                    signals[i] = -0.25
                    position = -1
                elif price <= donchian_low * 1.005:  # within 0.5% of lower band
                    signals[i] = 0.25
                    position = 1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                if is_trending:
                    # Exit long on breakdown below Donchian low in trend
                    if price < donchian_low:
                        exit_signal = True
                else:  # ranging
                    # Exit long when price moves to middle of range or hits upper band
                    if price >= donchian_high * 0.995:  # near upper band
                        exit_signal = True
            
            elif position == -1:  # short position
                if is_trending:
                    # Exit short on breakout above Donchian high in trend
                    if price > donchian_high:
                        exit_signal = True
                else:  # ranging
                    # Exit short when price moves to middle of range or hits lower band
                    if price <= donchian_low * 1.005:  # near lower band
                        exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Chop_Donchian_BreakoutMeanRev_Volume"
timeframe = "4h"
leverage = 1.0