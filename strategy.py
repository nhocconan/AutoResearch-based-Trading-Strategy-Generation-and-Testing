#!/usr/bin/env python3
# 6h_market_regime_adaptive_v1
# Hypothesis: On 6h timeframe, use a regime filter based on 1d ADX and Bollinger Band Width to adapt strategy:
# - In trending regimes (ADX > 25): trade breakouts of 12-period Donchian channels with volume confirmation
# - In ranging regimes (ADX <= 25 and BBW < median): trade mean reversion at Bollinger Bands (2,2) with volume confirmation
# - Uses volume filter (current volume > 1.5x 20-period average) to avoid false signals
# - Designed to work in both bull and bear markets by adapting to market conditions
# - Targets 50-150 total trades over 4 years (12-37/year) with strict entry conditions

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_market_regime_adaptive_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for regime filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = daily_high[1:] - daily_low[1:]
    tr2 = np.abs(daily_high[1:] - daily_close[:-1])
    tr3 = np.abs(daily_low[1:] - daily_close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((daily_high[1:] - daily_high[:-1]) > (daily_low[:-1] - daily_low[1:]), 
                       np.maximum(daily_high[1:] - daily_high[:-1], 0), 0)
    dm_minus = np.where((daily_low[:-1] - daily_low[1:]) > (daily_high[1:] - daily_high[:-1]), 
                        np.maximum(daily_low[:-1] - daily_low[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smooth(tr, 14)
    dm_plus_smooth = wilders_smooth(dm_plus, 14)
    dm_minus_smooth = wilders_smooth(dm_minus, 14)
    
    # DI and DX
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smooth(dx, 14)
    
    # Calculate 1d Bollinger Bands (20,2)
    sma_20 = pd.Series(daily_close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(daily_close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    bb_width = bb_upper - bb_lower
    
    # Median BBW for regime detection
    bbw_median = np.nanmedian(bb_width)
    
    # Align 1d indicators to 6h
    adx_6h = align_htf_to_ltf(prices, df_1d, adx)
    bbw_median_6h = align_htf_to_ltf(prices, df_1d, bbw_median * np.ones_like(bb_width))
    sma_20_6h = align_htf_to_ltf(prices, df_1d, sma_20)
    std_20_6h = align_htf_to_ltf(prices, df_1d, std_20)
    
    # Calculate 6-period Donchian channels on 6h
    def donchian_channels(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_hi, donch_lo = donchian_channels(high, low, 12)
    
    # Volume confirmation: 20-period average on 6h
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(adx_6h[i]) or np.isnan(bbw_median_6h[i]) or np.isnan(sma_20_6h[i]) or \
           np.isnan(std_20_6h[i]) or np.isnan(donch_hi[i]) or np.isnan(donch_lo[i]) or \
           np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Determine regime
        is_trending = adx_6h[i] > 25
        is_ranging = (adx_6h[i] <= 25) and (bb_width[i] < bbw_median_6h[i]) if i < len(bb_width) else False
        
        if position == 1:  # Long position
            # Exit conditions
            if is_trending:
                # Exit trend trade: price crosses below Donchian lower or volume drops
                if close[i] <= donch_lo[i] or volume[i] < avg_volume[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:
                # Exit range trade: price returns to mean or volume drops
                if close[i] >= sma_20_6h[i] or volume[i] < avg_volume[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions
            if is_trending:
                # Exit trend trade: price crosses above Donchian upper or volume drops
                if close[i] >= donch_hi[i] or volume[i] < avg_volume[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:
                # Exit range trade: price returns to mean or volume drops
                if close[i] <= sma_20_6h[i] or volume[i] < avg_volume[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            if is_trending and volume_ok:
                # Trend regime: breakout entries
                if close[i] > donch_hi[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < donch_lo[i]:
                    position = -1
                    signals[i] = -0.25
            elif is_ranging and volume_ok:
                # Range regime: mean reversion at Bollinger Bands
                bb_upper_6h = sma_20_6h[i] + 2 * std_20_6h[i]
                bb_lower_6h = sma_20_6h[i] - 2 * std_20_6h[i]
                if close[i] < bb_lower_6h:
                    position = 1
                    signals[i] = 0.25
                elif close[i] > bb_upper_6h:
                    position = -1
                    signals[i] = -0.25
    
    return signals