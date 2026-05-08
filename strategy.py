#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day volume profile to identify high volume nodes (HVN) as support/resistance.
# Combines: 1) Price rejection at HVN with volume confirmation, 2) 4h EMA(20) trend filter, 3) 1-day ATR volatility filter.
# HVN act as institutional reference points where large volume traded, making them significant S/R.
# In trending markets: pullbacks to HVN in direction of trend offer high-probability entries.
# In ranging markets: reversals at HVN offer mean-reversion opportunities.
# Designed for low trade frequency (<30/year) with strong edge in both bull and bear markets.

name = "4h_VolumeProfile_HVN_Rejection_TrendFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for volume profile construction
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Volume Profile: Price levels with most volume
    # Use 20-day lookback to build profile, update daily
    lookback = 20
    price_bins = 50  # Number of price levels for profile
    
    # Arrays to store HVN (High Volume Node) levels
    hvn_resistance = np.full(len(close_1d), np.nan)  # Resistance HVN above price
    hvn_support = np.full(len(close_1d), np.nan)    # Support HVN below price
    
    # Build volume profile for each day
    for i in range(lookback, len(close_1d)):
        # Get lookback period data
        start_idx = i - lookback
        end_idx = i
        
        # Price range for this period
        period_high = np.max(high_1d[start_idx:end_idx])
        period_low = np.min(low_1d[start_idx:end_idx])
        price_range = period_high - period_low
        
        if price_range <= 0:
            continue
            
        # Create price bins
        bin_edges = np.linspace(period_low, period_high, price_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Volume distribution across price bins
        volume_profile = np.zeros(price_bins)
        
        # Distribute each day's volume to price bins
        for j in range(start_idx, end_idx):
            # Typical price for the day
            typical_price = (high_1d[j] + low_1d[j] + close_1d[j]) / 3
            # Find which bin this price falls into
            bin_idx = np.searchsorted(bin_edges, typical_price) - 1
            bin_idx = max(0, min(bin_idx, price_bins - 1))
            volume_profile[bin_idx] += volume_1d[j]
        
        # Find the bin with maximum volume (Point of Control - POC)
        poc_idx = np.argmax(volume_profile)
        poc_price = bin_centers[poc_idx]
        
        # Find significant volume nodes (>= 70% of POC volume)
        volume_threshold = volume_profile[poc_idx] * 0.7
        significant_bins = volume_profile >= volume_threshold
        
        if np.any(significant_bins):
            # Get HVN above and below current price
            current_price = close_1d[i]
            above_bins = significant_bins & (bin_centers > current_price)
            below_bins = significant_bins & (bin_centers < current_price)
            
            if np.any(above_bins):
                # Closest HVN above as resistance
                hvn_resistance[i] = bin_centers[above_bins][np.argmin(bin_centers[above_bins] - current_price)]
            if np.any(below_bins):
                # Closest HVN below as support
                hvn_support[i] = bin_centers[below_bins][np.argmin(current_price - bin_centers[below_bins])]
    
    # First lookback days have no data
    hvn_resistance[:lookback] = np.nan
    hvn_support[:lookback] = np.nan
    
    # Align HVN levels to 4h timeframe
    hvn_resistance_aligned = align_htf_to_ltf(prices, df_1d, hvn_resistance)
    hvn_support_aligned = align_htf_to_ltf(prices, df_1d, hvn_support)
    
    # 4h EMA(20) for trend filter
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 1-day ATR for volatility filter (avoid choppy markets)
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    true_range = np.maximum(np.maximum(high_low, high_close), low_close)
    atr_1d = pd.Series(true_range).ewm(span=14, adjust=False, min_periods=14).mean().values
    # Normalize ATR by price to get volatility percentage
    volatility = atr_1d / close_1d
    # Only trade when volatility is reasonable (not too high, not too low)
    vol_filter = (volatility > 0.01) & (volatility < 0.08)  # 1% to 8% daily volatility
    
    # Align volatility filter to 4h
    vol_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_filter.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA(20) and alignment
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(hvn_resistance_aligned[i]) or np.isnan(hvn_support_aligned[i]) or
            np.isnan(ema_20[i]) or np.isnan(vol_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if volatility filter fails
        if vol_filter_aligned[i] < 0.5:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: price near support HVN with bullish rejection
            if hvn_support_aligned[i] > 0:
                # Price within 0.5% of support HVN
                price_to_support = abs(close[i] - hvn_support_aligned[i]) / close[i]
                if price_to_support < 0.005:
                    # Bullish rejection: close near high of bar
                    if close[i] > (high[i] + low[i]) / 2:
                        # Only take if above EMA20 (uptrend)
                        if close[i] > ema_20[i]:
                            signals[i] = 0.25
                            position = 1
            
            # Short setup: price near resistance HVN with bearish rejection
            elif hvn_resistance_aligned[i] > 0:
                # Price within 0.5% of resistance HVN
                price_to_resistance = abs(close[i] - hvn_resistance_aligned[i]) / close[i]
                if price_to_resistance < 0.005:
                    # Bearish rejection: close near low of bar
                    if close[i] < (high[i] + low[i]) / 2:
                        # Only take if below EMA20 (downtrend)
                        if close[i] < ema_20[i]:
                            signals[i] = -0.25
                            position = -1
        
        elif position == 1:
            # Long exit: price reaches resistance HVN or trend turns down
            if hvn_resistance_aligned[i] > 0:
                price_to_resistance = abs(close[i] - hvn_resistance_aligned[i]) / close[i]
                if price_to_resistance < 0.005:
                    signals[i] = 0.0
                    position = 0
            elif close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches support HVN or trend turns up
            if hvn_support_aligned[i] > 0:
                price_to_support = abs(close[i] - hvn_support_aligned[i]) / close[i]
                if price_to_support < 0.005:
                    signals[i] = 0.0
                    position = 0
            elif close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals