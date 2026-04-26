#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeRegime_v1
Hypothesis: Camarilla R1/S1 breakout on 12h timeframe with 1d EMA34 trend filter and volume spike (>2x median) to capture strong momentum moves. Uses choppiness regime filter to avoid sideways markets. Designed for 12h timeframe to target 12-37 trades/year with tight entry conditions requiring confluence of breakout, trend, volume, and regime filter. Works in bull/bear markets by only trading with the 1d trend and avoiding low-volume, choppy conditions.
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
    
    # Get 1d data for HTF trend (EMA34) and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels from previous 1d OHLC
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    camarilla_r1 = prev_close_1d + (1.0/6) * (prev_high_1d - prev_low_1d)
    camarilla_s1 = prev_close_1d - (1.0/6) * (prev_high_1d - prev_low_1d)
    
    # Align HTF indicators to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume spike filter: volume > 2x median volume (20-period) for conviction
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # ATR(14) for volatility-based stops
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Choppiness regime filter: avoid trading when CHOP > 61.8 (range/choppy market)
    # Calculate CHOP(14) = 100 * log10(sum(ATR(1),14) / (log10(highest_high(14) - lowest_low(14)) * sqrt(14)))
    atr_1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    atr_1[0] = high[0] - low[0]
    atr_sum = pd.Series(atr_1).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denom = (highest_high - lowest_low) * np.sqrt(14)
    # Avoid division by zero
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop = 100 * np.log10(atr_sum / chop_denom)
    chop = np.where(np.isnan(chop), 50.0, chop)  # neutral when not enough data
    
    # Choppiness regime: CHOP <= 61.8 = trending (good to trade), CHOP > 61.8 = ranging (avoid)
    chop_regime = chop <= 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of EMA(34) 1d, volume median (20), ATR (14), CHOP (14)
    start_idx = max(34, 20, 14, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_median[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(chop_regime[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_34_1d_val = ema_34_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_val = atr[i]
        chop_regime_val = chop_regime[i]
        
        # Trend filter: price > EMA34 (uptrend) or < EMA34 (downtrend)
        uptrend = close_val > ema_34_1d_val
        downtrend = close_val < ema_34_1d_val
        
        # Volume spike filter: only trade in above-average volume environments
        volume_spike = volume_val > 2.0 * vol_median_val
        
        if position == 0:
            # Long: break above R1 with volume spike, uptrend, and trending regime
            long_signal = (close_val > camarilla_r1_aligned[i]) and \
                          volume_spike and \
                          uptrend and \
                          chop_regime_val
            
            # Short: break below S1 with volume spike, downtrend, and trending regime
            short_signal = (close_val < camarilla_s1_aligned[i]) and \
                           volume_spike and \
                           downtrend and \
                           chop_regime_val
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, high_val)
            # ATR trailing stop
            if close_val < highest_since_entry - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, low_val)
            # ATR trailing stop
            if close_val > lowest_since_entry + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeRegime_v1"
timeframe = "12h"
leverage = 1.0