#!/usr/bin/env python3
name = "4h_KAMA_Direction_RSI20_Pullback_1dTrend"
timeframe = "4h"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # KAMA (Kaufman Adaptive Moving Average) on 4h close
    # ER = Efficiency Ratio = abs(close - close[10]) / sum(abs(diff(close, 10)))
    # SC = [ER * (fastest - slowest) + slowest]^2 where fastest=2/(2+1), slowest=2/(30+1)
    # KAMA = prev_KAMA + SC * (close - prev_KAMA)
    fast_sc = 2 / (2 + 1)  # 0.6667
    slow_sc = 2 / (30 + 1)  # 0.0645
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    
    # Calculate ER and SC for each point
    change = np.zeros(n)
    volatility = np.zeros(n)
    er = np.zeros(n)
    sc = np.zeros(n)
    
    for i in range(10, n):
        change[i] = abs(close[i] - close[i-10])
        volatility[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI(14) for pullback entries
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    rs = np.full(n, np.nan)
    rsi = np.full(n, 50.0)  # default neutral
    
    # Wilder's smoothing
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        
        if avg_loss[i] > 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 2  # ~8 hours for 4h to reduce trades
    
    start_idx = max(34, 20, 14, 10)  # ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine 1d trend direction
        trend_up = close > ema_34_1d_aligned[i]
        trend_down = close < ema_34_1d_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price pulls back to KAMA in uptrend with RSI < 30 and volume
            if (close[i] <= kama[i] * 1.005 and  # within 0.5% above KAMA
                close[i] >= kama[i] * 0.995 and  # within 0.5% below KAMA
                rsi[i] < 30 and
                trend_up[i] and
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price pulls back to KAMA in downtrend with RSI > 70 and volume
            elif (close[i] <= kama[i] * 1.005 and
                  close[i] >= kama[i] * 0.995 and
                  rsi[i] > 70 and
                  trend_down[i] and
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price moves away from KAMA or RSI overbought or trend changes
            if (close[i] > kama[i] * 1.02 or  # 2% above KAMA
                rsi[i] > 70 or
                not trend_up[i]):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price moves away from KAMA or RSI oversold or trend changes
            if (close[i] < kama[i] * 0.98 or   # 2% below KAMA
                rsi[i] < 30 or
                not trend_down[i]):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: KAMA direction with RSI pullback entries on 4h timeframe.
# Long when price pulls back to KAMA (within 0.5%) in 1d uptrend with RSI < 30 and volume confirmation.
# Short when price pulls back to KAMA (within 0.5%) in 1d downtrend with RSI > 70 and volume confirmation.
# Uses KAMA for adaptive trend following, RSI for mean-reversion entries, and 1d EMA34 for trend filter.
# Works in bull markets (buy pullbacks in uptrend) and bear markets (sell pullbacks in downtrend).
# Low trade frequency expected due to multiple conditions: KAMA proximity + RSI extreme + volume + trend alignment.