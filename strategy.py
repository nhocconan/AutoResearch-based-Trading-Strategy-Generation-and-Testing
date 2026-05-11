#!/usr/bin/env python3
name = "6h_Premium_Discount_Order_Block_v2"
timeframe = "6h"
leverage = 1.0

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
    
    # Premium/Discount zones from 12h timeframe (not 1d to avoid overlap with HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h mid-price (average of high and low)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    mid_12h = (high_12h + low_12h) / 2
    
    # Align 12h mid-price to 6h timeframe
    mid_12h_aligned = align_htf_to_ltf(prices, df_12h, mid_12h)
    
    # Calculate 12h ATR for zone width
    tr_12h = np.maximum(
        high_12h - low_12h,
        np.maximum(
            np.abs(high_12h - np.roll(close_12h, 1)),
            np.abs(low_12h - np.roll(close_12h, 1))
        )
    )
    # Need close_12h for TR calculation
    close_12h = df_12h['close'].values
    tr_12h = np.maximum(
        high_12h - low_12h,
        np.maximum(
            np.abs(high_12h - np.roll(close_12h, 1)),
            np.abs(low_12h - np.roll(close_12h, 1))
        )
    )
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Premium/Discount zones: ±0.5 * ATR from mid
    upper_zone = mid_12h_aligned + 0.5 * atr_12h_aligned
    lower_zone = mid_12h_aligned - 0.5 * atr_12h_aligned
    
    # Order blocks: significant price levels with volume imbalance
    # Identify swing points on 6h timeframe
    lookback = 5
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n - lookback):
        if high[i] == np.max(high[i-lookback:i+lookback+1]):
            swing_high[i] = True
        if low[i] == np.min(low[i-lookback:i+lookback+1]):
            swing_low[i] = True
    
    # Volume at swing points
    vol_swing_high = np.where(swing_high, volume, 0)
    vol_swing_low = np.where(swing_low, volume, 0)
    
    # Average volume at swing points for threshold
    avg_vol_swing_high = pd.Series(vol_swing_high).rolling(window=20, min_periods=5).mean().values
    avg_vol_swing_low = pd.Series(vol_swing_low).rolling(window=20, min_periods=5).mean().values
    
    # Significant order blocks: volume > 1.5x average at swing points
    ob_high = swing_high & (volume > 1.5 * avg_vol_swing_high)
    ob_low = swing_low & (volume > 1.5 * avg_vol_swing_low)
    
    # Store OB levels (most recent)
    ob_levels_high = np.full(n, np.nan)
    ob_levels_low = np.full(n, np.nan)
    
    last_ob_high = np.nan
    last_ob_low = np.nan
    
    for i in range(n):
        if ob_high[i]:
            last_ob_high = high[i]
        if ob_low[i]:
            last_ob_low = low[i]
        ob_levels_high[i] = last_ob_high
        ob_levels_low[i] = last_ob_low
    
    # Trend filter: 6h EMA 20
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.3 * vol_ma20
    
    signals = np.zeros(n)
    
    start_idx = max(20, 20)  # EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(mid_12h_aligned[i]) or np.isnan(atr_12h_aligned[i]) or np.isnan(ema_20[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions: price in discount zone, above EMA, at/below OB low, volume
        in_discount = close[i] <= lower_zone[i]
        above_ema = close[i] > ema_20[i]
        at_ob_low = ob_levels_low[i] > 0 and close[i] <= ob_levels_low[i] * 1.001  # within 0.1% of OB low
        
        # Short conditions: price in premium zone, below EMA, at/above OB high, volume
        in_premium = close[i] >= upper_zone[i]
        below_ema = close[i] < ema_20[i]
        at_ob_high = ob_levels_high[i] > 0 and close[i] >= ob_levels_high[i] * 0.999  # within 0.1% of OB high
        
        if in_discount and above_ema and (at_ob_low or not np.isnan(ob_levels_low[i])) and volume_filter[i]:
            signals[i] = 0.25
        elif in_premium and below_ema and (at_ob_high or not np.isnan(ob_levels_high[i])) and volume_filter[i]:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals