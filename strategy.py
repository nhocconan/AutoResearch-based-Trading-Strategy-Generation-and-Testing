#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ELDER_RAY_POWER_SMART_MONEY"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for Elder Ray calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate 13-period EMA on 1d close for Elder Ray
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = df_1d['high'].values - ema13_1d
    bear_power_1d = df_1d['low'].values - ema13_1d
    
    # Align Elder Ray components to 6h timeframe
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # 1d trend filter: EMA13 slope (positive = uptrend, negative = downtrend)
    ema13_slope_1d = np.diff(ema13_1d, prepend=ema13_1d[0])
    trend_1d = (ema13_slope_1d > 0).astype(float)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Smart Money Confirmation: Volume-Weighted Average Price deviation
    # Calculate VWAP on 6h data
    typical_price = (high + low + close) / 3.0
    vwap_num = pd.Series(typical_price * volume).rolling(window=24, min_periods=24).sum().values
    vwap_den = pd.Series(volume).rolling(window=24, min_periods=24).sum().values
    vwap = np.where(vwap_den != 0, vwap_num / vwap_den, typical_price)
    
    # Price position relative to VWAP: above VWAP = bullish bias, below = bearish
    price_vs_vwap = close - vwap
    
    # Volume confirmation: current volume > 1.5 * 24-period average
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_confirm = volume > (vol_ma24 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # warmup for VWAP and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or 
            np.isnan(trend_1d_aligned[i]) or np.isnan(vwap) or np.isnan(vol_ma24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Bull Power positive AND price above VWAP AND uptrend AND volume confirmation
            long_cond = (bull_power_6h[i] > 0 and 
                        price_vs_vwap[i] > 0 and 
                        trend_1d_aligned[i] > 0.5 and 
                        vol_confirm[i])
            
            # Short entry: Bear Power negative AND price below VWAP AND downtrend AND volume confirmation
            short_cond = (bear_power_6h[i] < 0 and 
                         price_vs_vwap[i] < 0 and 
                         trend_1d_aligned[i] < 0.5 and 
                         vol_confirm[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bear Power turns negative OR price crosses below VWAP
            if bear_power_6h[i] < 0 or price_vs_vwap[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull Power turns positive OR price crosses above VWAP
            if bull_power_6h[i] > 0 or price_vs_vwap[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Elder Ray Power (Bull/Bear) strategy with VWAP bias and trend filter on 6h timeframe.
# Uses 1d Elder Ray to identify institutional buying/selling pressure aligned with 6b candles.
# Enters long when Bull Power > 0 (buying pressure) + price above VWAP + uptrend + volume confirmation.
# Enters short when Bear Power < 0 (selling pressure) + price below VWAP + downtrend + volume confirmation.
# Exits when power shifts or price crosses VWAP, capturing institutional reversals.
# Works in bull markets (buying pressure dominance) and bear markets (selling pressure dominance).
# Volume confirmation ensures institutional participation, reducing false signals.
# Targets 15-25 trades/year on 6h timeframe (60-100 total over 4 years).