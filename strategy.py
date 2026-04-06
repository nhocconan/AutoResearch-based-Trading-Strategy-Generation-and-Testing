#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour volume-weighted price action with 1d support/resistance zones and 1w trend filter.
# Uses volume clusters to identify institutional accumulation/distribution zones.
# Enters on breakouts from these zones with volume confirmation and trend alignment.
# Works in bull markets (breakouts above accumulation zones) and bear markets (breakdowns below distribution zones).
# Target: 75-150 total trades over 4 years.

name = "exp_13372_12h_volume_cluster_breakout_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
VOLUME_CLUSTER_LOOKBACK = 20   # periods to identify volume clusters
VOLUME_THRESHOLD = 2.0         # volume must be 2x average to confirm breakout
CLUSTER_STRENGTH = 1.5         # minimum cluster strength to form a zone
SIGNAL_SIZE = 0.25             # 25% position size
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def find_volume_clusters(high, low, close, volume, lookback):
    """Find volume-weighted price clusters (high volume nodes)"""
    n = len(close)
    clusters_high = np.full(n, np.nan)
    clusters_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        # Get lookback window
        h_window = high[i-lookback:i]
        l_window = low[i-lookback:i]
        c_window = close[i-lookback:i]
        v_window = volume[i-lookback:i]
        
        # Create price bins
        price_min = np.min(l_window)
        price_max = np.max(h_window)
        if price_max <= price_min:
            continue
            
        n_bins = 20
        bin_edges = np.linspace(price_min, price_max, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Volume distribution
        volume_profile = np.zeros(n_bins)
        for j in range(len(v_window)):
            # Find which bin this candle's price range falls into
            weight = v_window[j]
            # Distribute volume across the candle's range
            low_price = l_window[j]
            high_price = h_window[j]
            
            # Simple approach: assign to bin based on midpoint
            mid_price = (low_price + high_price) / 2
            bin_idx = np.searchsorted(bin_edges, mid_price) - 1
            bin_idx = max(0, min(bin_idx, n_bins - 1))
            volume_profile[bin_idx] += weight
        
        # Find high volume nodes (clusters)
        if np.max(volume_profile) > 0:
            # Find peaks in volume profile
            volume_threshold = np.mean(volume_profile) * CLUSTER_STRENGTH
            peaks = []
            for j in range(1, n_bins - 1):
                if volume_profile[j] > volume_threshold and \
                   volume_profile[j] > volume_profile[j-1] and \
                   volume_profile[j] > volume_profile[j+1]:
                    peaks.append(bin_centers[j])
            
            if len(peaks) >= 2:
                clusters_high[i] = max(peaks)
                clusters_low[i] = min(peaks)
            elif len(peaks) == 1:
                clusters_high[i] = peaks[0]
                clusters_low[i] = peaks[0]
    
    return clusters_high, clusters_low

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for volume clusters
    df_1d = get_htf_data(prices, '1d')
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate daily volume clusters
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    cluster_high, cluster_low = find_volume_clusters(
        high_1d, low_1d, close_1d, volume_1d, VOLUME_CLUSTER_LOOKBACK
    )
    
    # Align clusters to 12h timeframe
    cluster_high_aligned = align_htf_to_ltf(prices, df_1d, cluster_high)
    cluster_low_aligned = align_htf_to_ltf(prices, df_1d, cluster_low)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(20, VOLUME_CLUSTER_LOOKBACK) + 1
    
    for i in range(start, n):
        # Skip if data not available
        if np.isnan(ema_1w_aligned[i]) or np.isnan(cluster_high_aligned[i]) or np.isnan(cluster_low_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/below weekly EMA
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # Breakout signals from volume clusters
        # Only trade if clusters are valid (not equal and not NaN)
        valid_clusters = not np.isnan(cluster_high_aligned[i-1]) and not np.isnan(cluster_low_aligned[i-1]) and \
                         cluster_high_aligned[i-1] != cluster_low_aligned[i-1]
        
        breakout_up = False
        breakout_down = False
        
        if valid_clusters:
            breakout_up = volume_ok and uptrend and (high[i] > cluster_high_aligned[i-1])
            breakout_down = volume_ok and downtrend and (low[i] < cluster_low_aligned[i-1])
        
        # Generate signals
        if position == 0:
            if breakout_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_down:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals