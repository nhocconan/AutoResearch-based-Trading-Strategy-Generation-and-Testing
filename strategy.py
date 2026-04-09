#!/usr/bin/env python3
# 1h_rsi_mean_reversion_4h1d_filter_v1
# Hypothesis: On 1h timeframe, use RSI(14) for mean reversion entries when RSI < 30 (long) or RSI > 70 (short).
# Filter trades with 4h trend (price > 20-period EMA for long, price < 20-period EMA for short) and 1d regime (chop < 61.8 for ranging markets only).
# Session filter: 08-20 UTC to avoid low-volume Asian session noise.
# Discrete position sizing: 0.20 to limit fee drag. Target: 15-37 trades/year (60-150 over 4 years).
# Works in bull markets via 4h trend filter and in bear markets via 1d chop regime (mean reversion in ranging markets).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi_mean_reversion_4h1d_filter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # RSI(14) on 1h close
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral RSI when insufficient data
    
    # 4h HTF trend filter: 20-period EMA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    ema_20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d HTF regime filter: Chopiness Index (14) for ranging markets
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    
    # ATR(14)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chopiness Index: 100 * log10(sum(atr)/log(hh-ll)) / log10(14)
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    hh_ll = hh - ll
    # Avoid division by zero or log of zero
    chop = np.where((hh_ll > 0) & (sum_atr > 0), 
                    100 * np.log10(sum_atr) / np.log10(14) / np.log10(hh_ll), 
                    50.0)  # neutral when invalid
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN or outside session
        if (np.isnan(rsi[i]) or np.isnan(ema_20_4h_aligned[i]) or np.isnan(chop_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI returns to neutral (50) or 4h trend turns bearish
            if rsi[i] >= 50 or close[i] < ema_20_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral (50) or 4h trend turns bullish
            if rsi[i] <= 50 or close[i] > ema_20_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter only in ranging market (chop > 61.8) with RSI extreme and 4h trend alignment
            if chop_aligned[i] > 61.8:  # Ranging market regime
                # Long: RSI oversold and 4h trend bullish
                if rsi[i] < 30 and close[i] > ema_20_4h_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Short: RSI overbought and 4h trend bearish
                elif rsi[i] > 70 and close[i] < ema_20_4h_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals