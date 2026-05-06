#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1w EMA34 trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 AND Bear Power rising (less negative) AND price > 1w EMA34 (uptrend) AND volume spike
# Short when Bear Power < 0 AND Bull Power falling (less positive) AND price < 1w EMA34 (downtrend) AND volume spike
# Uses discrete sizing 0.25 to balance profit potential and fee drag; target 50-150 total trades over 4 years (12-37/year)
# Works in both bull/bear: trend filter avoids counter-trend traps, volume filter ensures participation, Elder Ray captures momentum shifts

name = "6h_ElderRay_1wEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 trend filter
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate EMA13 for Elder Ray (on 6h data)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Calculate smoothed Elder Ray for momentum confirmation
    bull_power_smooth = pd.Series(bull_power).ewm(span=5, adjust=False, min_periods=5).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Calculate ATR(14) for stoploss
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume spike filter (>2.0x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_20)
    
    # Align HTF indicators to 6h timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0
    short_extreme = 0.0
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(bull_power_smooth[i]) or 
            np.isnan(bear_power_smooth[i]) or np.isnan(atr[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        if position == 0:
            # Long entry: Bull Power > 0 AND Bear Power rising (less negative) AND uptrend AND volume spike
            if (bull_power[i] > 0 and bear_power_smooth[i] > bear_power_smooth[i-1] and 
                close[i] > ema34_1w_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
                long_extreme = close[i]
            # Short entry: Bear Power < 0 AND Bull Power falling (less positive) AND downtrend AND volume spike
            elif (bear_power[i] < 0 and bull_power_smooth[i] < bull_power_smooth[i-1] and 
                  close[i] < ema34_1w_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
                short_extreme = close[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, close[i])
            # Exit long: price retraces 30% of ATR from extreme
            if close[i] <= long_extreme - 0.3 * atr[i]:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update short extreme
            short_extreme = min(short_extreme, close[i])
            # Exit short: price retraces 30% of ATR from extreme
            if close[i] >= short_extreme + 0.3 * atr[i]:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals