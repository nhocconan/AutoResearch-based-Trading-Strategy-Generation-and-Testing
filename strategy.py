#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction + RSI(14) + chop filter (CHOP > 61.8) for mean reversion in ranging markets
# - Primary signal: KAMA(10,2,30) direction - long when rising, short when falling
# - Entry filter: RSI(14) < 40 for long, > 60 for short (avoid strong trends)
# - Regime filter: Choppiness Index(14) > 61.8 (ranging market) - avoids false signals in strong trends
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 7-25 trades/year (30-100 total over 4 years) per 1d strategy guidelines
# - Works in bull/bear: KAMA adapts to volatility, RSI prevents entries in strong trends, chop filter ensures ranging conditions

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w EMA200 for trend filter (only used for regime, not direct signal)
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Pre-compute KAMA on 1d timeframe
    close = prices['close'].values
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # 10-period volatility
    # Avoid division by zero
    er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan, dtype=float)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Pre-compute RSI(14) on 1d timeframe
    delta = np.diff(close, n=1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain, dtype=float), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    # Handle first 14 values
    rsi[:14] = 50  # neutral start
    
    # Pre-compute Choppiness Index(14) on 1d timeframe
    high = prices['high'].values
    low = prices['low'].values
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Chop = 100 * log10(sum(ATR14) / (HH14 - LL14)) / log10(14)
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    hh_ll = highest_high - lowest_low
    chop = np.divide(
        100 * np.log10(sum_atr / hh_ll),
        np.log10(14),
        out=np.full_like(sum_atr, 50.0, dtype=float),
        where=(hh_ll != 0) & (~np.isnan(sum_atr)) & (~np.isnan(hh_ll))
    )
    chop[:13] = 50.0  # not enough data
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_200_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: KAMA falls OR RSI > 60 (overbought) OR chop < 38.2 (trending market)
            if (kama[i] < kama[i-1] or 
                rsi[i] > 60 or 
                chop[i] < 38.2):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: KAMA rises OR RSI < 40 (oversold) OR chop < 38.2 (trending market)
            if (kama[i] > kama[i-1] or 
                rsi[i] < 40 or 
                chop[i] < 38.2):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for KAMA direction with RSI and chop filters
            # Long: KAMA rising AND RSI < 40 AND chop > 61.8 (ranging market)
            if (kama[i] > kama[i-1] and 
                rsi[i] < 40 and 
                chop[i] > 61.8):
                position = 1
                signals[i] = 0.25
            # Short: KAMA falling AND RSI > 60 AND chop > 61.8 (ranging market)
            elif (kama[i] < kama[i-1] and 
                  rsi[i] > 60 and 
                  chop[i] > 61.8):
                position = -1
                signals[i] = -0.25
    
    return signals