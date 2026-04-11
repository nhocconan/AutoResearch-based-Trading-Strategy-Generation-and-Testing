#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_elder_ray_power_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate daily EMA13 and EMA20 for Elder Ray
    close_1d = df_1d['close'].values
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power = high - ema13  # Daily high minus EMA13
    bear_power = low - ema20   # Daily low minus EMA20
    
    # Align Elder Ray indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13)
    ema20_aligned = align_htf_to_ltf(prices, df_1d, ema20)
    
    # Calculate 60-period EMA for trend filter (on 6h)
    ema60 = pd.Series(close).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # Calculate ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    for i in range(40, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema60[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr_val > 0.008 * price_close  # ATR > 0.8% of price
        
        # Trend filter: price above/below EMA60
        uptrend = price_close > ema60[i]
        downtrend = price_close < ema60[i]
        
        # Elder Ray conditions
        strong_bull = bull_power_aligned[i] > 0 and bull_power_aligned[i] > -bear_power_aligned[i]
        strong_bear = bear_power_aligned[i] < 0 and bear_power_aligned[i] < bull_power_aligned[i]
        
        # Long conditions: uptrend + bull power dominance + volume + volatility
        long_signal = uptrend and strong_bull and volume_confirmed and vol_filter
        
        # Short conditions: downtrend + bear power dominance + volume + volatility
        short_signal = downtrend and strong_bear and volume_confirmed and vol_filter
        
        # Exit when power weakens or trend changes
        exit_long = position == 1 and (bull_power_aligned[i] <= 0 or not uptrend)
        exit_short = position == -1 and (bear_power_aligned[i] >= 0 or not downtrend)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Elder Ray Power with trend and volume filters on 6h timeframe.
# Uses daily Bull Power (high-EMA13) and Bear Power (low-EMA20) to measure
# bull/bear strength relative to trend. Enters long when Bull Power positive
# and dominant, price above 60-period EMA, with volume and volatility confirmation.
# Enters short when Bear Power negative and dominant, price below EMA60,
# with volume and volatility confirmation. Exits when power weakens or trend changes.
# Works in both bull and bear markets by measuring underlying strength/weakness.
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost.
# Elder Ray is effective in trending markets and avoids whipsaws in ranging conditions.