#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with RSI mean reversion and chop regime filter
# - Uses 1d timeframe to minimize trade frequency and fee drag
# - KAMA identifies adaptive trend direction (long when price > KAMA, short when price < KAMA)
# - RSI(14) provides mean reversion entries within the trend (long when RSI < 30, short when RSI > 70)
# - Choppiness Index (CHOP) filter avoids whipsaws in ranging markets (only trade when CHOP < 50 = trending)
# - Weekly HTF trend filter ensures alignment with major trend (only trade in direction of weekly EMA20)
# - Targets 7-25 trades/year (30-100 total over 4 years) to avoid fee drag
# - Works in both bull and bear markets: trend filter captures major moves, RSI provides mean reversion entries within trend

name = "1d_kama_rsi_chop_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Pre-compute 1d KAMA(14, 2, 30) for trend direction
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=1))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0) if len(change) > 0 else np.array([0.0])
    # Proper ER calculation for each point
    er = np.zeros_like(close)
    for i in range(1, len(close)):
        if i >= 1:
            price_change = np.abs(close[i] - close[i-14]) if i >= 14 else np.abs(close[i] - close[0])
            sum_abs_changes = np.sum(np.abs(np.diff(close[max(0, i-13):i+1]))) if i >= 1 else 0.0
            er[i] = price_change / (sum_abs_changes + 1e-10)
    
    # Smoothing constants
    fastest = 2.0 / (2 + 1)      # EMA(2)
    slowest = 2.0 / (30 + 1)     # EMA(30)
    sc = (er * (fastest - slowest) + slowest) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Pre-compute RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # Wilder's smoothing
    avg_gain[13] = np.mean(gain[1:14]) if len(gain) >= 14 else 0
    avg_loss[13] = np.mean(loss[1:14]) if len(loss) >= 14 else 0
    
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = 50  # Neutral before enough data
    
    # Pre-compute Choppiness Index(14)
    atr = np.zeros_like(close)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[0.0], tr])  # First TR is 0
    
    # ATR(14) using Wilder's smoothing
    atr[13] = np.mean(tr[1:14]) if len(tr) >= 14 else 0
    for i in range(14, len(close)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate highest high and lowest low over 14 periods
    hh = np.zeros_like(close)
    ll = np.zeros_like(close)
    for i in range(len(close)):
        start_idx = max(0, i-13)
        hh[i] = np.max(high[start_idx:i+1])
        ll[i] = np.min(low[start_idx:i+1])
    
    # Choppiness Index
    chop = np.zeros_like(close)
    for i in range(13, len(close)):
        if hh[i] > ll[i] and atr[i] > 0:
            sum_tr = np.sum(tr[max(0, i-12):i+1])
            chop[i] = 100 * np.log10(sum_tr / (hh[i] - ll[i])) / np.log10(14)
        else:
            chop[i] = 50  # Neutral when undefined
    chop[:13] = 50
    
    # Pre-compute volume average for confirmation
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price > KAMA (uptrend) AND RSI < 30 (oversold) AND CHOP < 50 (trending) AND weekly uptrend
            # Volume confirmation: > 1.2x average volume
            if (close[i] > kama[i] and 
                rsi[i] < 30 and 
                chop[i] < 50 and 
                close[i] > ema20_1w_aligned[i] and
                prices['volume'].iloc[i] > (1.2 * volume_20_avg[i])):
                position = 1
                signals[i] = 0.25
            # Short entry: price < KAMA (downtrend) AND RSI > 70 (overbought) AND CHOP < 50 (trending) AND weekly downtrend
            elif (close[i] < kama[i] and 
                  rsi[i] > 70 and 
                  chop[i] < 50 and 
                  close[i] < ema20_1w_aligned[i] and
                  prices['volume'].iloc[i] > (1.2 * volume_20_avg[i])):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. RSI returns to neutral territory (40-60)
            # 2. Price crosses KAMA (trend change)
            # 3. Chop > 60 (choppy/ranging market)
            if position == 1:  # Long position
                if (rsi[i] > 40 and rsi[i] < 60) or \
                   (close[i] < kama[i]) or \
                   (chop[i] > 60):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (rsi[i] > 40 and rsi[i] < 60) or \
                   (close[i] > kama[i]) or \
                   (chop[i] > 60):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals