#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + ADX trend filter with 1d regime
# - Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# - Long when Bull Power > 0 AND Bear Power < 0 AND ADX > 25 (trending up)
# - Short when Bear Power > 0 AND Bull Power < 0 AND ADX > 25 (trending down)
# - Use 1d chop regime: only trade when CHOP < 38.2 (trending) to avoid whipsaws in ranging markets
# - Discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Elder Ray measures bull/bear power relative to EMA
# - ADX filters for trending conditions
# - 1d chop regime ensures we only trade when higher timeframe is trending

name = "6h_1d_elder_ray_adx_chop_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # ADX(14)
    # True Range
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - np.roll(close, 1)[1:])
    tr3 = np.abs(low[1:] - np.roll(close, 1)[1:])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM-
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx = np.where((di_plus + di_minus) == 0, 0, adx)  # avoid division by zero
    
    # Pre-compute 1d chop regime (choppiness index)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1_1d = np.abs(high_1d[1:] - low_1d[1:])
    tr2_1d = np.abs(high_1d[1:] - np.roll(close_1d, 1)[1:])
    tr3_1d = np.abs(low_1d[1:] - np.roll(close_1d, 1)[1:])
    tr_1d = np.maximum(np.maximum(tr1_1d, tr2_1d), tr3_1d)
    tr_1d = np.concatenate([[np.nan], tr_1d])
    
    # ATR(14) for 1d
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    tr_sum_1d = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Max(high) - Min(low) over 14 periods
    max_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_max_min_1d = max_high_1d - min_low_1d
    
    # Chop = 100 * log10(tr_sum / range_max_min) / log10(14)
    chop_1d = 100 * np.log10(tr_sum_1d / range_max_min_1d) / np.log10(14)
    chop_1d = np.concatenate([np.full(13, np.nan), chop_1d[13:]])  # align indices
    
    # Chop regime: < 38.2 = trending (good for trend following)
    chop_trending = chop_1d < 38.2
    
    # Align HTF indicators to 6h timeframe
    chop_trending_aligned = align_htf_to_ltf(prices, df_1d, chop_trending)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx[i]) or np.isnan(chop_trending_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Bull Power > 0 AND Bear Power < 0 AND ADX > 25 AND 1d chop trending
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                adx[i] > 25 and 
                chop_trending_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Bear Power > 0 AND Bull Power < 0 AND ADX > 25 AND 1d chop trending
            elif (bear_power[i] > 0 and 
                  bull_power[i] < 0 and 
                  adx[i] > 25 and 
                  chop_trending_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when Elder Ray signals weaken (power crosses zero) OR ADX < 20 (trend weakening)
            exit_long = (position == 1 and 
                        (bull_power[i] <= 0 or bear_power[i] >= 0 or adx[i] < 20))
            exit_short = (position == -1 and 
                         (bear_power[i] <= 0 or bull_power[i] >= 0 or adx[i] < 20))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals