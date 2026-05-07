# Solution: 6h_HeikinAshi_Engulfing_Candle_with_1w_Trend_Filter
# Hypothesis: Heikin-Ashi smoothing reduces noise on 6H, and engulfing candles at key 1W trend extremes capture reversals with high probability.
# Works in bull/bear by using 1W trend direction to filter long/short signals only in direction of higher timeframe trend.
# Uses volume confirmation to avoid false signals. Target: 15-30 trades/year.

#!/usr/bin/env python3
name = "6h_HeikinAshi_Engulfing_Candle_with_1w_Trend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_heikin_ashi(open_price, high, low, close):
    """Calculate Heikin-Ashi candles."""
    ha_close = (open_price + high + low + close) / 4
    ha_open = np.zeros_like(close)
    ha_open[0] = (open_price[0] + close[0]) / 2
    for i in range(1, len(close)):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
    ha_high = np.maximum(np.maximum(high, low), np.maximum(ha_open, ha_close))
    ha_low = np.minimum(np.minimum(high, low), np.minimum(ha_open, ha_close))
    return ha_open, ha_high, ha_low, ha_close

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Load 1w and 1d data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 10 or len(df_1d) < 20:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1d volume average for spike detection
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate Heikin-Ashi for 6H data
    ha_open, ha_high, ha_low, ha_close = calculate_heikin_ashi(open_price, high, low, close)
    
    # Bullish engulfing: current HA green candle engulfs previous red candle
    bullish_engulf = (ha_close > ha_open) & (ha_open < ha_close) & \
                     (ha_close[i-1] < ha_open[i-1]) & (ha_close > ha_open[i-1]) & \
                     (ha_open < ha_close[i-1])
    # Bearish engulfing: current HA red candle engulfs previous green candle
    bearish_engulf = (ha_close < ha_open) & (ha_open > ha_close) & \
                     (ha_close[i-1] > ha_open[i-1]) & (ha_close < ha_open[i-1]) & \
                     (ha_open > ha_close[i-1])
    
    # Vectorized engulfing detection
    bullish_engulf = (ha_close > ha_open) & (ha_open < ha_close) & \
                     (np.roll(ha_close, 1) < np.roll(ha_open, 1)) & \
                     (ha_close > np.roll(ha_open, 1)) & \
                     (ha_open < np.roll(ha_close, 1))
    bearish_engulf = (ha_close < ha_open) & (ha_open > ha_close) & \
                     (np.roll(ha_close, 1) > np.roll(ha_open, 1)) & \
                     (ha_close < np.roll(ha_open, 1)) & \
                     (ha_open > np.roll(ha_close, 1))
    
    # Handle first element
    bullish_engulf[0] = False
    bearish_engulf[0] = False
    
    # Volume spike: current volume > 2.0x 1d average volume
    vol_spike = volume > 2.0 * vol_ma_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for EMA50 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bullish engulfing + price above 1w EMA50 (uptrend) + volume spike
            if (bullish_engulf[i] and close[i] > ema50_1w_aligned[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish engulfing + price below 1w EMA50 (downtrend) + volume spike
            elif (bearish_engulf[i] and close[i] < ema50_1w_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bearish engulfing or price below 1w EMA50
            if bearish_engulf[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bullish engulfing or price above 1w EMA50
            if bullish_engulf[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Note: Uses Heikin-Ashi smoothed candles to reduce noise, engulfing patterns for entry,
# 1W EMA50 for trend filter, and volume spike for confirmation. Targets 15-30 trades/year.