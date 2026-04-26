#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_ChopRegime_TrendFilter
Hypothesis: Combine TRIX momentum with volume spike (>2.0x 20-period average) and choppiness regime filter (CHOP > 61.8 for mean reversion, < 38.2 for trend) to capture high-probability reversals in ranging markets and continuations in trending markets. Use 1d EMA200 as long-term trend filter. Discrete sizing 0.25. Target 20-30 trades/year to minimize fee drag while adapting to BTC/ETH bull/bear/range regimes.
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
    open_time = prices['open_time'].values
    
    # Session filter: UTC 8-20 for institutional activity
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA200 for long-term trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # TRIX(12,9,9) - triple smoothed EMA rate of change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = 100 * (ema3.diff() / ema3.shift(1))
    trix_values = trix.values
    trix_signal = pd.Series(trix_values).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix_hist = trix_values - trix_signal  # TRIX histogram
    
    # Choppiness Index(14) - measures whether market is choppy (ranging) or trending
    # CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    # Handle division by zero or invalid values
    chop = np.where((highest_high - lowest_low) > 0, chop, 50.0)  # neutral when no range
    
    # Volume average (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of TRIX (12+9+9=30), CHOP (14), EMA200 (200), volume MA (20)
    start_idx = max(30, 14, 200, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(trix_hist[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or
            not in_session[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        trix_hist_val = trix_hist[i]
        chop_val = chop[i]
        ema_200_1d_val = ema_200_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average (strict to reduce trades)
        volume_confirmed = vol_val > 2.0 * vol_ma_val
        
        if position == 0:
            # Determine regime: choppy (mean revert) or trending (trend follow)
            is_choppy = chop_val > 61.8
            is_trending = chop_val < 38.2
            
            if is_choppy:
                # In choppy regime: mean reversion at extremes
                # Long: TRIX histogram crosses above -0.1 (oversold bounce) + volume + long-term uptrend
                long_signal = (trix_hist_val > -0.1) and (trix_hist[i-1] <= -0.1) and volume_confirmed and (close_val > ema_200_1d_val)
                # Short: TRIX histogram crosses below +0.1 (overbought rejection) + volume + long-term downtrend
                short_signal = (trix_hist_val < 0.1) and (trix_hist[i-1] >= 0.1) and volume_confirmed and (close_val < ema_200_1d_val)
            elif is_trending:
                # In trending regime: trend continuation
                # Long: TRIX histogram crosses above zero + volume + price above EMA200
                long_signal = (trix_hist_val > 0) and (trix_hist[i-1] <= 0) and volume_confirmed and (close_val > ema_200_1d_val)
                # Short: TRIX histogram crosses below zero + volume + price below EMA200
                short_signal = (trix_hist_val < 0) and (trix_hist[i-1] >= 0) and volume_confirmed and (close_val < ema_200_1d_val)
            else:
                # Transition regime: wait for clearer signal
                long_signal = False
                short_signal = False
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: TRIX histogram crosses below zero (momentum loss) OR close below EMA200 (trend break)
            if (trix_hist_val < 0 and trix_hist[i-1] >= 0) or (close_val < ema_200_1d_val):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: overextended in choppy regime (TRIX > +0.2)
            elif chop_val > 61.8 and trix_hist_val > 0.2:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: TRIX histogram crosses above zero (momentum loss) OR close above EMA200 (trend break)
            if (trix_hist_val > 0 and trix_hist[i-1] <= 0) or (close_val > ema_200_1d_val):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: overextended in choppy regime (TRIX < -0.2)
            elif chop_val > 61.8 and trix_hist_val < -0.2:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "4h_TRIX_VolumeSpike_ChopRegime_TrendFilter"
timeframe = "4h"
leverage = 1.0