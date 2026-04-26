#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI(14) for momentum confirmation, and Choppiness Index for regime filtering. 
Long when KAMA slope positive, RSI > 50, and CHOP < 61.8 (trending market).
Short when KAMA slope negative, RSI < 50, and CHOP < 61.8.
Avoids whipsaws in ranging markets (CHOP > 61.8) and false momentum signals.
Uses discrete sizing (0.25) to minimize fee drag. Target: 30-80 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === KAUFMAN ADAPTIVE MOVING AVERAGE (KAMA) ===
    # Fast EMA: 2 periods, Slow EMA: 30 periods
    # ER = |net change| / sum(|abs changes|) over lookback
    # Smooth = [ER * (fastest - slowest) + slowest]^2
    er_period = 10
    fast_sc = 2 / (2 + 1)   # 2-period EMA
    slow_sc = 2 / (30 + 1)  # 30-period EMA
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = pd.Series(change).rolling(window=er_period, min_periods=er_period).sum().values
    net_change = np.abs(np.diff(close, prepend=close[0]))  # Actually need cumulative net change
    # Recalculate net_change properly
    net_change_calc = np.zeros(n)
    for i in range(er_period, n):
        net_change_calc[i] = np.abs(close[i] - close[i-er_period])
    net_change = net_change_calc
    
    er = np.where(volatility > 0, net_change / volatility, 0)
    er[np.isnan(er)] = 0
    
    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA slope (trend direction)
    kama_slope = np.diff(kama, prepend=0)
    
    # === RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[np.isnan(rsi)] = 50  # Neutral when undefined
    
    # === CHOPPINESS INDEX (14) ===
    # CHOP > 61.8 = ranging (avoid trend trades), CHOP < 38.2 = trending
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]  # first bar
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero and invalid values
    range_hl = max_high - min_low
    chop = np.where((atr14 > 0) & (range_hl > 0), 
                    100 * np.log10(range_hl / atr14) / np.log10(14), 50)
    chop[np.isnan(chop)] = 50
    
    # === HTF: 1w trend filter (optional, for extra confirmation) ===
    # Use 1w EMA20 as higher timeframe trend bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 20:
        close_1w = df_1w['close'].values
        ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
        ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
        # 1w trend: price above/below 1w EMA20
        htf_trend_bias = close > ema_20_1w_aligned
    else:
        htf_trend_bias = np.ones(n, dtype=bool)  # No bias if insufficient data
    
    # === SIGNAL LOGIC ===
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup period (need 14 for RSI/CHOP, 10 for ER)
    start_idx = max(14, 10)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_slope[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or np.isnan(htf_trend_bias[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long conditions:
        # 1. KAMA slope positive (uptrend)
        # 2. RSI > 50 (bullish momentum)
        # 3. CHOP < 61.8 (trending market, not ranging)
        # 4. 1w EMA20 bias aligned (optional confirmation)
        long_condition = (kama_slope[i] > 0) and (rsi[i] > 50) and (chop[i] < 61.8) and htf_trend_bias[i]
        
        # Short conditions:
        # 1. KAMA slope negative (downtrend)
        # 2. RSI < 50 (bearish momentum)
        # 3. CHOP < 61.8 (trending market, not ranging)
        # 4. 1w EMA20 bias aligned (optional confirmation)
        short_condition = (kama_slope[i] < 0) and (rsi[i] < 50) and (chop[i] < 61.8) and htf_trend_bias[i]
        
        # Exit conditions: opposite signal or regime change to ranging
        exit_long = (kama_slope[i] <= 0) or (rsi[i] <= 50) or (chop[i] >= 61.8)
        exit_short = (kama_slope[i] >= 0) or (rsi[i] >= 50) or (chop[i] >= 61.8)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0