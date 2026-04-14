#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volatility regime filter and 1w trend confirmation.
# Long when price breaks above Donchian(20) high AND volatility regime is expanding (ATR ratio > 1.2) AND 1w EMA is rising.
# Short when price breaks below Donchian(20) low AND volatility regime is expanding AND 1w EMA is falling.
# Exit when price returns to Donchian midpoint or volatility contracts (ATR ratio < 0.8).
# Designed to capture volatility breakouts in trending markets while avoiding false signals in low volatility.
# Target: 25-30 trades/year per symbol (100-120 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    lookback = 20
    donch_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donch_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # ATR for volatility regime (14-period)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR ratio: current ATR / 50-period average ATR (volatility regime)
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / atr_ma
    
    # Load 1w data ONCE for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA(21) on 1w
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    # EMA slope: current EMA - previous EMA
    ema_slope = np.diff(ema_1w, prepend=np.nan)
    
    # Align indicators to lower timeframe
    donch_high_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low, 'close': close}), donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low, 'close': close}), donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low, 'close': close}), donch_mid)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    ema_slope_aligned = align_htf_to_ltf(prices, df_1w, ema_slope)
    atr_ratio_aligned = align_htf_to_ltf(prices, pd.DataFrame({'atr': atr}), atr_ratio)
    
    # Volume confirmation: 1.3x average volume (moderate filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(lookback, 50, 20)  # Need Donchian, ATR ratio, and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or
            np.isnan(donch_mid_aligned[i]) or
            np.isnan(ema_1w_aligned[i]) or
            np.isnan(ema_slope_aligned[i]) or
            np.isnan(atr_ratio_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime: expanding (ATR ratio > 1.2) for entry, contracting (< 0.8) for exit
        vol_expanding = atr_ratio_aligned[i] > 1.2
        vol_contracting = atr_ratio_aligned[i] < 0.8
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # Trend filter: EMA slope positive for uptrend, negative for downtrend
        uptrend = ema_slope_aligned[i] > 0
        downtrend = ema_slope_aligned[i] < 0
        
        if position == 0:
            # Look for Donchian breakouts with volatility expansion and trend
            # Long: price breaks above Donchian high AND volatility expanding AND uptrend
            if (close[i] > donch_high_aligned[i] and 
                vol_expanding and 
                uptrend and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low AND volatility expanding AND downtrend
            elif (close[i] < donch_low_aligned[i] and 
                  vol_expanding and 
                  downtrend and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian mid OR volatility contracts OR trend weakens
            if (close[i] <= donch_mid_aligned[i] or 
                vol_contracting or 
                ema_slope_aligned[i] <= 0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Donchian mid OR volatility contracts OR trend weakens
            if (close[i] >= donch_mid_aligned[i] or 
                vol_contracting or 
                ema_slope_aligned[i] >= 0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_VolRegime_1wEMA_v1"
timeframe = "4h"
leverage = 1.0