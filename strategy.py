#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout + 1w EMA50 trend filter + volume confirmation
# Camarilla pivots identify key support/resistance levels where price often reverses or breaks out
# 1w EMA50 ensures we trade with the higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation (1.5x 20-period average) filters weak breakouts
# Works in bull/bear: EMA50 trend filter avoids ranging markets, Camarilla levels work in all regimes
# Target: 30-100 total trades over 4 years (7-25/year) with discrete sizing 0.25-0.30

name = "1d_1w_camarilla_ema50_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend direction
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h2 = np.full(n, np.nan)
    camarilla_l2 = np.full(n, np.nan)
    camarilla_h1 = np.full(n, np.nan)
    camarilla_l1 = np.full(n, np.nan)
    camarilla_pivot = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous day's OHLC to calculate today's Camarilla levels
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        pivot = (prev_high + prev_low + prev_close) / 3.0
        range_hl = prev_high - prev_low
        
        camarilla_pivot[i] = pivot
        camarilla_h4[i] = pivot + range_hl * 1.1 / 2.0
        camarilla_l4[i] = pivot - range_hl * 1.1 / 2.0
        camarilla_h3[i] = pivot + range_hl * 1.1 / 4.0
        camarilla_l3[i] = pivot - range_hl * 1.1 / 4.0
        camarilla_h2[i] = pivot + range_hl * 1.1 / 6.0
        camarilla_l2[i] = pivot - range_hl * 1.1 / 6.0
        camarilla_h1[i] = pivot + range_hl * 1.1 / 12.0
        camarilla_l1[i] = pivot - range_hl * 1.1 / 12.0
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_pivot[i]) or np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < Camarilla L3 OR price < 1w EMA50 (trend change)
            if close[i] < camarilla_l3[i] or close[i] < ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Camarilla H3 OR price > 1w EMA50 (trend change)
            if close[i] > camarilla_h3[i] or close[i] > ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Camarilla breakout + 1w EMA50 trend filter
            if volume_confirmed:
                # Long entry: price > Camarilla H4 AND price > 1w EMA50 (bullish breakout + uptrend)
                if close[i] > camarilla_h4[i] and close[i] > ema_50_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < Camarilla L4 AND price < 1w EMA50 (bearish breakout + downtrend)
                elif close[i] < camarilla_l4[i] and close[i] < ema_50_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals