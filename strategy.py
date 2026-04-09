#!/usr/bin/env python3
# 6h_keltner_donchian_breakout_v1
# Hypothesis: 6h strategy combining Donchian breakout direction with Keltner channel mean reversion
# for filtering false breakouts. Long when price breaks above Donchian(20) AND closes above
# Keltner upper band (strong momentum). Short when price breaks below Donchian(20) AND closes
# below Keltner lower band (strong downside momentum). Uses volume confirmation (>1.3x average)
# to ensure breakout validity. Daily HTF trend filter: only trade in direction of daily EMA(50).
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 25-35 trades/year.
# Works in bull markets via breakout continuation and in bear markets via strong downside breaks.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_keltner_donchian_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily HTF data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for daily EMA(50)
        return np.zeros(n)
    
    close_d = df_1d['close'].values
    # Daily EMA(50) for trend filter
    ema_50_d = pd.Series(close_d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_d)
    
    # 6h Donchian(20) channels
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donch_high = high_s.rolling(window=20, min_periods=20).max().values
    donch_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # 6h Keltner Channel (20, 2.0)
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    tp_s = pd.Series(typical_price)
    ema_tp = tp_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR(20) for Keltner width
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_s = pd.Series(tr)
    atr = atr_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    keltner_upper = ema_tp + 2.0 * atr
    keltner_lower = ema_tp - 2.0 * atr
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or
            np.isnan(volume_ma[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        # Daily trend filter: only trade long if price > daily EMA(50), short if price < daily EMA(50)
        trend_filter_long = close[i] > ema_50_aligned[i]
        trend_filter_short = close[i] < ema_50_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price falls below Donchian low OR loses volume confirmation
            if close[i] < donch_low[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above Donchian high OR loses volume confirmation
            if close[i] > donch_high[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Long entry: price breaks above Donchian high AND closes above Keltner upper
                # AND daily trend supports long (price > daily EMA50)
                if (close[i] > donch_high[i] and 
                    close[i] > keltner_upper[i] and 
                    trend_filter_long):
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below Donchian low AND closes below Keltner lower
                # AND daily trend supports short (price < daily EMA50)
                elif (close[i] < donch_low[i] and 
                      close[i] < keltner_lower[i] and 
                      trend_filter_short):
                    position = -1
                    signals[i] = -0.25
    
    return signals