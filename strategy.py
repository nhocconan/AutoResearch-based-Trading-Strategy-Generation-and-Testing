#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and 1w chop regime filter
# - Long when price breaks above Donchian(20) high with volume > 1.3x 20-period EMA and 1d chop > 61.8 (range)
# - Short when price breaks below Donchian(20) low with volume > 1.3x 20-period EMA and 1d chop > 61.8 (range)
# - Exit: ATR trailing stop (2.5x ATR) or Donchian(10) opposite break
# - Position sizing: 0.25 discrete level
# - Targets ~25-35 trades/year on 4h timeframe. Donchian breakouts capture momentum,
#   volume confirmation validates breakout strength, chop filter avoids false breakouts in ranging markets.
#   Works in bull/bear: breakouts work in both regimes, chop filter ensures we only trade when range-bound conditions prevail.

name = "4h_1d_1w_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Calculate Donchian channels on 4h
    donchian_period = 20
    donchian_exit_period = 10
    
    highest_high_20 = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low_20 = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    highest_high_10 = pd.Series(high).rolling(window=donchian_exit_period, min_periods=donchian_exit_period).max().values
    lowest_low_10 = pd.Series(low).rolling(window=donchian_exit_period, min_periods=donchian_exit_period).min().values
    
    # Calculate 1d volume EMA for confirmation (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    volume_ema_20_1d = pd.Series(volume_1d).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ema_20_1d)
    
    # Calculate 1d Chopiness Index(14) for regime filter
    # Chop = 100 * log10(sum(atr(14)) / (log10(highest_high - lowest_low) * 14)) / log10(14)
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = 0  # First TR is 0 as there's no previous close
    
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    highest_high_14d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero and log of zero/negative
    hh_ll_diff = highest_high_14d - lowest_low_14d
    hh_ll_diff = np.where(hh_ll_diff <= 0, 1e-10, hh_ll_diff)  # Small positive value if zero or negative
    
    sum_atr_14 = atr_1d
    chop_denominator = np.log10(hh_ll_diff) * 14
    chop_denominator = np.where(chop_denominator <= 0, 1e-10, chop_denominator)  # Avoid log<=0
    
    chopiness = 100 * (np.log10(sum_atr_14) - np.log10(chop_denominator)) / np.log10(14)
    chopiness = np.where(np.isnan(chopiness) | np.isinf(chopiness), 50, chopiness)  # Default to neutral
    chopiness_aligned = align_htf_to_ltf(prices, df_1d, chopiness)
    
    # Calculate ATR(14) for trailing stop on 4h
    tr1_4h = high - low
    tr2_4h = np.abs(high - np.roll(close, 1))
    tr3_4h = np.abs(low - np.roll(close, 1))
    tr2_4h[0] = tr2_4h[1] if len(tr2_4h) > 1 else 0
    tr3_4h[0] = tr3_4h[1] if len(tr3_4h) > 1 else 0
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_4h = pd.Series(tr_4h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(volume_ema_20_1d_aligned[i]) or np.isnan(chopiness_aligned[i]) or 
            np.isnan(atr_4h[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume for confirmation (aligned to 4h)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm = vol_1d_current[i] > 1.3 * volume_ema_20_1d_aligned[i]
        
        # Regime filter: 1d chop > 61.8 indicates ranging market (good for breakout fade avoidance)
        regime_filter = chopiness_aligned[i] > 61.8
        
        # Donchian breakout entry conditions
        # Long: price breaks above Donchian(20) high
        # Short: price breaks below Donchian(20) low
        long_entry = (close[i] > highest_high_20[i] and 
                     vol_confirm and 
                     regime_filter)
        short_entry = (close[i] < lowest_low_20[i] and 
                      vol_confirm and 
                      regime_filter)
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
            elif short_entry:
                position = -1
                signals[i] = -0.25
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Update highest/lowest since entry
            if position == 1:
                highest_since_entry = max(highest_since_entry, close[i])
                lowest_since_entry = min(lowest_since_entry, close[i])
                # ATR trailing stop: exit if price drops 2.5*ATR from high
                # or Donchian(10) break down
                if (close[i] < highest_since_entry - 2.5 * atr_4h[i] or  # trailing stop
                    close[i] < lowest_low_10[i]):         # Donchian(10) break down
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                highest_since_entry = max(highest_since_entry, close[i])
                lowest_since_entry = min(lowest_since_entry, close[i])
                # ATR trailing stop: exit if price rises 2.5*ATR from low
                # or Donchian(10) break up
                if (close[i] > lowest_since_entry + 2.5 * atr_4h[i] or  # trailing stop
                    close[i] > highest_high_10[i]):         # Donchian(10) break up
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals