#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI + chop regime filter
# KAMA (Kaufman Adaptive Moving Average) adapts to market noise - fast in trends, slow in chop
# RSI(14) for momentum confirmation with discrete thresholds
# Choppiness Index regime filter: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trending (trend follow)
# Uses 1w EMA(34) for stronger trend alignment to reduce whipsaw
# Designed for very low trade frequency (7-25/year) to minimize fee drag on 1d timeframe
# Works in both bull and bear markets by adapting to regime

name = "1d_KAMA_RSI_Chop_1wEMA34_v1"
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
    
    # Get 1w data for EMA(34) trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1w for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA to 1d timeframe (wait for completed 1w bar)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # KAMA (Kaufman Adaptive Moving Average) on 1d
    # Efficiency Ratio: ER = |net change| / sum(|changes|)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = pd.Series(change).rolling(window=10, min_periods=10).sum().values
    net_change = np.abs(np.diff(close, prepend=close[0]))
    for i in range(10, len(net_change)):
        net_change[i] = np.abs(close[i] - close[i-10])
    er = np.divide(net_change, volatility, out=np.zeros_like(net_change), where=volatility!=0)
    # Smoothing constants: fast = 2/(2+1), slow = 2/(30+1)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # Start after ER period
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI(14) on 1d
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14) on 1d
    # True Range
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - np.roll(close, 1))
    tr3 = np.abs(np.roll(low, 1) - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Sum of True Range over period
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over period
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Choppiness Index
    chop = np.divide(
        np.log10(atr_sum) * np.log(10),
        np.log10(hh - ll) * np.log(14) + np.log10(atr_sum) * np.log(10),
        out=np.full_like(atr_sum, 50.0),
        where=(hh - ll) > 0
    )
    chop = 100 * chop
    
    # Volume confirmation (1.5x 20-period average) on 1d
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = 50  # max(10 for KAMA ER, 14 for RSI/chop, 20 for volume MA +1 for shift, 34 for 1w EMA)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Bullish: price > KAMA + RSI > 50 + chop < 38.2 (trending) + above 1w EMA(34) + volume spike
            if (close[i] > kama[i] and rsi[i] > 50 and chop[i] < 38.2 and 
                close[i] > ema_34_1w_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Bearish: price < KAMA + RSI < 50 + chop < 38.2 (trending) + below 1w EMA(34) + volume spike
            elif (close[i] < kama[i] and rsi[i] < 50 and chop[i] < 38.2 and 
                  close[i] < ema_34_1w_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < KAMA OR RSI < 40 OR chop > 61.8 (choppy) OR below 1w EMA(34)
            if (close[i] < kama[i] or rsi[i] < 40 or chop[i] > 61.8 or 
                close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > KAMA OR RSI > 60 OR chop > 61.8 (choppy) OR above 1w EMA(34)
            if (close[i] > kama[i] or rsi[i] > 60 or chop[i] > 61.8 or 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals