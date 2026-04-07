#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Weekly KAMA Trend with Volume and Chop Filter
# Hypothesis: KAMA adapts to market noise - in trending markets it follows price closely,
# in ranging markets it stays flat. Combined with weekly trend filter (price vs weekly KAMA),
# volume confirmation, and choppy market filter (Choppiness Index > 61.8), this strategy
# captures strong trends while avoiding whipsaws in ranging markets. Works in both bull and bear:
# - In bull: price above weekly KAMA + low chop = uptrend, go long
# - In bear: price below weekly KAMA + low chop = downtrend, go short
# Uses volume to confirm institutional participation. Target: 10-25 trades/year (40-100 over 4 years).

name = "1d_weekly_kama_trend_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and chop calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Weekly close for KAMA and chop calculation
    weekly_close = df_weekly['close'].values
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Calculate Kaufman Adaptive Moving Average (KAMA) on weekly data
    # ER = |Change| / Volatility, where Change = |close - close[10]|, Volatility = sum|diff| over 10 periods
    change = np.abs(weekly_close - np.roll(weekly_close, 10))
    volatility = np.zeros_like(weekly_close)
    for i in range(10, len(weekly_close)):
        volatility[i] = np.sum(np.abs(np.diff(weekly_close[i-9:i+1])))
    
    # Avoid division by zero
    er = np.zeros_like(weekly_close)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(weekly_close)
    kama[0] = weekly_close[0]
    for i in range(1, len(weekly_close)):
        kama[i] = kama[i-1] + sc[i] * (weekly_close[i] - kama[i-1])
    
    # Calculate Choppiness Index on weekly data
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(n)
    atr_weekly = np.zeros_like(weekly_close)
    tr = np.zeros_like(weekly_close)
    for i in range(1, len(weekly_close)):
        tr[i] = max(
            weekly_high[i] - weekly_low[i],
            abs(weekly_high[i] - weekly_close[i-1]),
            abs(weekly_low[i] - weekly_close[i-1])
        )
    
    # Calculate ATR(14) - simplified as average TR
    atr_period = 14
    for i in range(atr_period, len(weekly_close)):
        atr_weekly[i] = np.mean(tr[i-atr_period+1:i+1])
    
    chop = np.zeros_like(weekly_close)
    lookback = 14
    for i in range(lookback, len(weekly_close)):
        sum_atr = np.sum(atr_weekly[i-lookback+1:i+1])
        max_high = np.max(weekly_high[i-lookback+1:i+1])
        min_low = np.min(weekly_low[i-lookback+1:i+1])
        range_val = max_high - min_low
        if range_val > 0:
            chop[i] = 100 * np.log10(sum_atr / range_val) / np.log10(lookback)
        else:
            chop[i] = 50  # neutral when no range
    
    # Align weekly indicators to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_weekly, kama)
    chop_aligned = align_htf_to_ltf(prices, df_weekly, chop)
    
    # Volume filter: volume > 1.3x 20-day average (more relaxed for daily)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(kama_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Chop filter: only trade when market is NOT choppy (CHOP < 61.8 = trending)
        if chop_aligned[i] > 61.8:
            # In choppy markets, go flat
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below KAMA or volume drops
            if close[i] <= kama_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price crosses above KAMA or volume drops
            if close[i] >= kama_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price above weekly KAMA with volume (uptrend)
            if close[i] > kama_aligned[i] and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short: price below weekly KAMA with volume (downtrend)
            elif close[i] < kama_aligned[i] and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals