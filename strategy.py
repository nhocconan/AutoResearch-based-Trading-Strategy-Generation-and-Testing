#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Keltner Channel Breakout with 1w EMA Trend Filter and Volume Spike
# Uses Keltner Channel (20, 1.5*ATR) for volatility-based breakout entries
# 1w EMA (50) provides multi-timeframe trend direction to avoid counter-trend trades
# Volume confirmation (>1.8x average) ensures institutional participation
# Designed to work in both bull and bear markets by trading breakouts in direction of higher timeframe trend
# Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w EMA data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Keltner Channel (20, 1.5*ATR)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    close_series = pd.Series(close)
    ma_20 = close_series.rolling(window=20, min_periods=20).mean().values
    upper_keltner = ma_20 + 1.5 * atr
    lower_keltner = ma_20 - 1.5 * atr
    
    # Volume confirmation: volume > 1.8x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for 1w EMA and Keltner calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(ma_20[i]) or 
            np.isnan(atr[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade in direction of 1w EMA
        trend_up = price > ema_50_1w_aligned[i]
        trend_down = price < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Keltner with volume filter and uptrend
            if price > upper_keltner[i] and vol > 1.8 * avg_vol[i] and trend_up:
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower Keltner with volume filter and downtrend
            elif price < lower_keltner[i] and vol > 1.8 * avg_vol[i] and trend_down:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below middle line (mean reversion)
            if price < ma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above middle line (mean reversion)
            if price > ma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_WeeklyKeltner_Breakout_1wEMA_Volume"
timeframe = "1d"
leverage = 1.0