#!/usr/bin/env python3
"""
Experiment #2754: 1h RSI mean reversion with 4h trend filter and volume confirmation
HYPOTHESIS: In ranging markets (common in bear/transition), RSI extremes + volume spike
provide high-probability mean reversion entries. 4h trend filter prevents counter-trend
trades, reducing whipsaws. Session filter (08-20 UTC) avoids low-liquidity hours.
Target: 60-150 total trades over 4 years = 15-37/year for 1h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2754_1h_rsi_meanrev_4h_trend_vol"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === HTF: 4h data for trend filter ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # 4h EMA(50) for trend
    ema_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_4h = np.where(close_4h > ema_4h, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # === 1h Indicators: RSI(14), Volume MA(20) ===
    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    # Wilder's smoothing (alpha = 1/period)
    alpha = 1.0 / 14
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[14] = np.mean(gain[1:15])  # seed
    avg_loss[14] = np.mean(loss[1:15])
    for i in range(15, n):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # For first 14 periods, RSI undefined -> set to 50 (neutral)
    rsi[:14] = 50.0
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # sufficient for RSI and volume MA
    
    for i in range(warmup, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any indicator invalid
        if (np.isnan(trend_4h_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Exit conditions: RSI returns to neutral range (40-60) or opposite extreme
            if position_side > 0:  # Long
                if rsi[i] >= 40 and rsi[i] <= 60:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                elif rsi[i] > 70:  # Overbought - take profit
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                if rsi[i] >= 40 and rsi[i] <= 60:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                elif rsi[i] < 30:  # Oversold - take profit
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 4h trend alignment for bias filter
        trend_bias = trend_4h_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: RSI oversold (<30) with uptrend on 4h
            if trend_bias > 0 and rsi[i] < 30:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short entry: RSI overbought (>70) with downtrend on 4h
            elif trend_bias < 0 and rsi[i] > 70:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals