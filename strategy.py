#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d ATR-based volatility breakout with volume confirmation and 1w trend filter.
# In volatile markets, price often breaks out of ATR ranges with strong volume.
# Uses 1d ATR(14) to define dynamic breakout levels: upper = close + 1.5*ATR, lower = close - 1.5*ATR.
# Enters long when price breaks above upper band with volume > 2x 24-period average.
# Enters short when price breaks below lower band with volume > 2x 24-period average.
# Uses 1w EMA(50) as trend filter: only long when above EMA, only short when below EMA.
# Exits when price returns to the 1d close or trend changes.
# Designed for low frequency: volatility breakouts are rare events, targeting 15-25 trades/year.
# Works in bull/bear: volatility expansions occur in both regimes.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on daily
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR is just high-low
    atr_14 = np.full(len(tr), np.nan)
    for i in range(14, len(tr)):
        if i == 14:
            atr_14[i] = np.mean(tr[1:15])  # seed with first 14
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Calculate dynamic bands: upper = prev_close + 1.5*ATR, lower = prev_close - 1.5*ATR
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = np.nan
    upper_band_1d = prev_close_1d + 1.5 * atr_14
    lower_band_1d = prev_close_1d - 1.5 * atr_14
    
    # Align daily bands to 12h
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band_1d)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band_1d)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close_1d)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: volume > 2 x 24-period average (12h)
    vol_ma_24 = np.full(n, np.nan)
    for i in range(23, n):
        vol_ma_24[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need daily ATR (30), weekly EMA (50), volume MA (24)
    start_idx = max(30, 50, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or
            np.isnan(prev_close_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_24[i]
        
        # Volume filter
        vol_filter = vol_now > 2.0 * vol_avg
        
        # Trend filters
        weekly_bullish = price > ema_50_1w_aligned[i]
        weekly_bearish = price < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper band with volume and weekly bullish
            if price > upper_band_aligned[i] and vol_filter and weekly_bullish:
                signals[i] = size
                position = 1
            # Short: price breaks below lower band with volume and weekly bearish
            elif price < lower_band_aligned[i] and vol_filter and weekly_bearish:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to previous close or weekly trend turns bearish
            if price < prev_close_aligned[i] or not weekly_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to previous close or weekly trend turns bullish
            if price > prev_close_aligned[i] or not weekly_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_ATR_VolatilityBreakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0