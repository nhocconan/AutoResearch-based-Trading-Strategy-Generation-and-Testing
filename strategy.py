# Solution
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h chart with 12h Keltner Channel breakout and 1d volume confirmation.
# In trending markets, price breaks the Keltner channel with volume; in ranging markets,
# price reverts to the EMA middle band. Uses 1d trend filter to avoid counter-trend trades.
# Keltner Channel: middle = EMA(20), upper/lower = EMA ± 2*ATR(10).
# Target: 20-30 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for Keltner Channel calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h EMA(20) for middle band
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    # 12h ATR(10) for channel width
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10_12h = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Keltner Channel bands
    upper_12h = ema_20_12h + 2.0 * atr_10_12h
    lower_12h = ema_20_12h - 2.0 * atr_10_12h
    
    # Align Keltner Channel to 6h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    middle_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Load 1d data for trend filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume ratio (current / 20-period average)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = vol_1d / np.where(vol_ma_20_1d == 0, 1, vol_ma_20_1d)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # 6h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(middle_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper = upper_aligned[i]
        lower = lower_aligned[i]
        middle = middle_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_ratio_1d = vol_ratio_1d_aligned[i]
        
        # Trend filter: only trade in direction of 1d trend
        uptrend = price > ema_trend
        downtrend = price < ema_trend
        
        # Volume filter: require above-average 1d volume
        vol_filter = vol_ratio_1d > 1.3
        
        if position == 0:
            # Look for long when price breaks above upper channel with volume in uptrend
            if uptrend and vol_filter:
                if price > upper:
                    signals[i] = 0.25
                    position = 1
            # Look for short when price breaks below lower channel with volume in downtrend
            elif downtrend and vol_filter:
                if price < lower:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price returns to middle band or trend reverses
            if price <= middle or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle band or trend reverses
            if price >= middle or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_12h_KeltnerBreakout_1dTrendVolumeFilter_v1"
timeframe = "6h"
leverage = 1.0