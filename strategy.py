# 1d Keltner Channel Breakout with 1W Trend Filter and Volume Confirmation
# Hypothesis: Breakouts from Keltner Channel (20, 2.0) on daily timeframe, filtered by 1-week EMA trend and volume spikes, work in both bull and bear markets by capturing momentum after volatility contraction. Low-frequency signals reduce fee drag while maintaining edge.

#!/usr/bin/env python3
name = "1d_KeltnerBreakout_1wEMA_Trend_Volume"
timeframe = "1d"
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
    
    # Get 1W data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 1W EMA34 trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Keltner Channel (20, 2.0) on daily data
    atr_period = 20
    atr = np.full(n, np.nan)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first TR
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, n):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    ma_20 = np.full(n, np.nan)
    ma_20[atr_period-1] = np.mean(close[:atr_period])
    for i in range(atr_period, n):
        ma_20[i] = (ma_20[i-1] * (atr_period-1) + close[i]) / atr_period
    
    kc_upper = ma_20 + 2.0 * atr
    kc_lower = ma_20 - 2.0 * atr
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    vol_ma_20[19] = np.mean(volume[:20])
    for i in range(20, n):
        vol_ma_20[i] = (vol_ma_20[i-1] * 19 + volume[i]) / 20
    vol_filter = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 3  # 3 days to reduce trades
    
    start_idx = max(100, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(kc_upper[i]) or 
            np.isnan(kc_lower[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine 1W trend direction
        trend_up = close > ema_34_1w_aligned[i]
        trend_down = close < ema_34_1w_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price breaks above Keltner upper with volume in uptrend
            if (close[i] > kc_upper[i] and 
                trend_up[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below Keltner lower with volume in downtrend
            elif (close[i] < kc_lower[i] and 
                  trend_down[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls below Keltner middle or trend changes
            if close[i] < ma_20[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises above Keltner middle or trend changes
            if close[i] > ma_20[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Keltner Channel breakout with 1-week EMA trend filter and volume confirmation on daily timeframe.
# Long when price breaks above upper Keltner band (20, 2.0) in uptrend with volume spike.
# Short when price breaks below lower Keltner band in downtrend with volume spike.
# Uses daily timeframe for low trade frequency (target: 30-100 total trades over 4 years).
# Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend).
# Volume confirmation ensures breakouts are genuine, not false signals.