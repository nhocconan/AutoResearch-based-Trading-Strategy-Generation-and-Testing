# 4h_HeikinAshi_Trend_HeikenAsi_Offset_Signal_v1
# Hypothesis: Heikin-Ashi candles filter out market noise and reveal true trend direction.
# Using HA close > HA open for uptrend and HA close < HA open for downtrend, combined with
# volume confirmation and ADX trend strength filter. Works in both bull and bear markets
# by following the dominant trend on 4H timeframe with strict entry conditions to limit trades.
# Target: 20-40 trades/year per symbol to minimize fee drag.

#!/usr/bin/env python3
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
    
    # Calculate Heikin-Ashi candles
    ha_close = (open_prices + high + low + close) / 4
    ha_open = np.zeros_like(close)
    ha_open[0] = (open_prices[0] + close[0]) / 2
    for i in range(1, n):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
    ha_high = np.maximum.reduce([high, ha_open, ha_close])
    ha_low = np.minimum.reduce([low, ha_open, ha_close])
    
    # Calculate ADX for trend strength (using 14 periods)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period]) if not np.any(np.isnan(data[:period])) else np.nan
        for i in range(period, len(data)):
            if np.isnan(result[i-1]) or np.isnan(data[i]):
                result[i] = np.nan
            else:
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period_adx = 14
    tr_smooth = wilders_smoothing(tr, period_adx)
    plus_di_smooth = wilders_smoothing(plus_dm, period_adx)
    minus_di_smooth = wilders_smoothing(minus_dm, period_adx)
    
    # Avoid division by zero
    dx = np.zeros(n)
    mask = tr_smooth > 0
    dx[mask] = 100 * np.abs(plus_di_smooth[mask] - minus_di_smooth[mask]) / tr_smooth[mask]
    adx = wilders_smoothing(dx, period_adx)
    
    # Volume filter: 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma20 + 1e-10)
    
    # Heikin-Ashi trend signals
    ha_uptrend = ha_close > ha_open
    ha_downtrend = ha_close < ha_open
    
    # Combined signals
    long_signal = ha_uptrend & (adx > 25) & (vol_ratio > 1.5)
    short_signal = ha_downtrend & (adx > 25) & (vol_ratio > 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ha_open[i]) or np.isnan(ha_close[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        if long_signal[i] and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_signal[i] and position >= 0:
            signals[i] = -0.25
            position = -1
        elif not ha_uptrend[i] and position == 1:
            signals[i] = 0.0
            position = 0
        elif not ha_downtrend[i] and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_HeikinAshi_Trend_HeikenAsi_Offset_Signal_v1"
timeframe = "4h"
leverage = 1.0