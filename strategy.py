#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume spike confirmation.
# Uses 1d ATR(14) percentile to identify low volatility regimes (chop) where breakouts are more reliable.
# Long when: price breaks above Donchian upper channel AND volume > 1.5x 20-period MA AND 1d ATR(14) < 30th percentile.
# Short when: price breaks below Donchian lower channel AND volume > 1.5x 20-period MA AND 1d ATR(14) < 30th percentile.
# ATR regime filter ensures we only trade breakouts during low volatility, reducing false breakouts in choppy markets.
# Works in bull (breakouts continue) and bear (breakdowns continue) by following price structure.
# Target: 12-25 trades/year to minimize fee drag while capturing strong moves.

name = "12h_Donchian20_1dATRRegime_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # ATR(14)
    atr_1d = np.full_like(close_1d, np.nan)
    for i in range(14, len(tr)):
        if not np.isnan(atr_1d[i-1]):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
        else:
            atr_1d[i] = np.nanmean(tr[i-13:i+1]) if i >= 13 else np.nan
    
    # Calculate 30th percentile of ATR(1d) for regime filter (low volatility)
    # Use expanding window to avoid look-ahead
    atr_percentile_30 = np.full_like(atr_1d, np.nan)
    for i in range(50, len(atr_1d)):  # need sufficient history
        valid_atr = atr_1d[:i+1][~np.isnan(atr_1d[:i+1])]
        if len(valid_atr) >= 20:
            atr_percentile_30[i] = np.percentile(valid_atr, 30)
    
    # Align 1d ATR percentile to 12h
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile_30)
    
    # Donchian channels (20-period) on 12h data
    lookback = 20
    highest_high = np.full_like(close, np.nan)
    lowest_low = np.full_like(close, np.nan)
    
    for i in range(lookback-1, len(close)):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume spike confirmation: volume > 1.5x 20-period MA
    volume_ma = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        volume_ma[i] = np.mean(volume[i-20:i])
    
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 20, 50)  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(atr_regime_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_vol_ma = volume_ma[i]
        curr_atr_regime = atr_regime_aligned[i]
        
        # Regime filter: only trade when ATR is below 30th percentile (low volatility)
        low_volatility_regime = curr_atr_regime > 0 and curr_atr_regime < np.percentile(
            atr_1d[~np.isnan(atr_1d)][:min(i+1, len(atr_1d))], 30) if i < len(atr_1d) else False
        
        # Simplified regime check: use pre-computed percentile
        if i < len(atr_regime_aligned) and not np.isnan(atr_regime_aligned[i]):
            # Get historical ATR values up to current point for percentile calculation
            hist_atr = atr_1d[:min(i+1, len(atr_1d))]
            valid_hist_atr = hist_atr[~np.isnan(hist_atr)]
            if len(valid_hist_atr) >= 20:
                regime_threshold = np.percentile(valid_hist_atr, 30)
                low_volatility_regime = curr_atr_regime < regime_threshold
            else:
                low_volatility_regime = False
        else:
            low_volatility_regime = False
        
        # Volume confirmation
        vol_confirm = curr_volume > (curr_vol_ma * 1.5) if not np.isnan(curr_vol_ma) else False
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: break above upper channel + volume spike + low volatility regime
            if (curr_close > highest_high[i] and 
                vol_confirm and 
                low_volatility_regime):
                signals[i] = 0.25
                position = 1
            # Short: break below lower channel + volume spike + low volatility regime
            elif (curr_close < lowest_low[i] and 
                  vol_confirm and 
                  low_volatility_regime):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below lower channel OR ATR regime shifts to high volatility
            if (curr_close < lowest_low[i] or 
                (i < len(atr_regime_aligned) and not np.isnan(atr_regime_aligned[i]) and
                 curr_atr_regime > np.percentile(
                     atr_1d[~np.isnan(atr_1d)][:min(i+1, len(atr_1d))], 70) if i < len(atr_1d) else False)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above upper channel OR ATR regime shifts to high volatility
            if (curr_close > highest_high[i] or 
                (i < len(atr_regime_aligned) and not np.isnan(atr_regime_aligned[i]) and
                 curr_atr_regime > np.percentile(
                     atr_1d[~np.isnan(atr_1d)][:min(i+1, len(atr_1d))], 70) if i < len(atr_1d) else False)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals