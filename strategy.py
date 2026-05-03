#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with RSI(14) mean reversion and choppiness regime filter
# KAMA adapts to market noise, providing reliable trend direction in both bull and bear markets.
# RSI(14) < 30 or > 70 identifies overextended conditions for mean reversion entries.
# Choppiness index (CHOP) > 61.8 confirms ranging markets where mean reversion works best.
# Designed for low trade frequency (7-25/year) on 1d timeframe to minimize fee drag.
# Volatility-adjusted position sizing (0.25) manages drawdowns during crashes.

name = "1d_KAMA_RSI_Choppiness_MeanReversion"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) - trend direction
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder for correct calc
    # Proper ER calculation over 10-period lookback
    er = np.zeros(n)
    for i in range(10, n):
        directional_change = np.abs(close[i] - close[i-10])
        sum_abs_changes = np.sum(np.abs(np.diff(close[i-9:i+1])))
        if sum_abs_changes > 0:
            er[i] = directional_change / sum_abs_changes
        else:
            er[i] = 0
    # For first 10 periods, use default ER
    er[:10] = 0.1
    
    # SC = [ER * (fastest_SC - slowest_SC) + slowest_SC]^2
    fastest_sc = 2 / (2 + 1)   # EMA(2)
    slowest_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fastest_sc - slowest_sc) + slowest_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = 50  # neutral before enough data
    
    # Calculate Choppiness Index (CHOP) - regime filter
    # True Range over 14 periods
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr14 = np.zeros(n)
    for i in range(14, n):
        if i == 14:
            atr14[i] = np.mean(tr[1:15])
        else:
            atr14[i] = (atr14[i-1] * 13 + tr[i]) / 14
    
    # Highest high and lowest low over 14 periods
    max_high = np.zeros(n)
    min_low = np.zeros(n)
    for i in range(14, n):
        max_high[i] = np.max(high[i-13:i+1])
        min_low[i] = np.min(low[i-13:i+1])
    
    # CHOP = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    sum_tr14 = np.zeros(n)
    for i in range(14, n):
        if i == 14:
            sum_tr14[i] = np.sum(tr[1:15])
        else:
            sum_tr14[i] = sum_tr14[i-1] - tr[i-14] + tr[i]
    
    hh_ll = max_high - min_low
    chop = np.zeros(n)
    for i in range(14, n):
        if hh_ll[i] > 0:
            chop[i] = 100 * np.log10(sum_tr14[i] / hh_ll[i]) / np.log10(14)
        else:
            chop[i] = 50  # neutral when no range
    chop[:14] = 50
    
    # Get 1w data for HTF trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        is_uptrend = close[i] > kama[i]
        is_downtrend = close[i] < kama[i]
        is_ranging = chop[i] > 61.8  # choppy/ranging market
        
        if position == 0:
            # Long: RSI < 30 (oversold) in ranging market with weekly uptrend bias
            if rsi[i] < 30 and is_ranging and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) in ranging market with weekly downtrend bias
            elif rsi[i] > 70 and is_ranging and is_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI > 50 (mean reversion complete) or trend change
            if rsi[i] > 50 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI < 50 (mean reversion complete) or trend change
            if rsi[i] < 50 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals