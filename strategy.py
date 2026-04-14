#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data (HTF) once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 12:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ATR (14-period)
    tr = np.zeros(len(df_1w))
    tr[0] = high_1w[0] - low_1w[0]
    for i in range(1, len(df_1w)):
        tr[i] = max(
            high_1w[i] - low_1w[i],
            abs(high_1w[i] - close_1w[i-1]),
            abs(low_1w[i] - close_1w[i-1])
        )
    
    atr_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 14:
        atr_1w[13] = np.mean(tr[:14])
        for i in range(14, len(df_1w)):
            atr_1w[i] = (atr_1w[i-1] * 13 + tr[i]) / 14
    
    # Calculate weekly ATR percentage (ATR / close)
    atr_pct_1w = np.full(len(df_1w), np.nan)
    for i in range(len(df_1w)):
        if not np.isnan(atr_1w[i]) and close_1w[i] > 0:
            atr_pct_1w[i] = atr_1w[i] / close_1w[i]
        else:
            atr_pct_1w[i] = np.nan
    
    # Calculate weekly volatility filter (top 30% volatility weeks)
    vol_threshold_1w = np.full(len(df_1w), np.nan)
    for i in range(len(df_1w)):
        if i >= 12:  # Need 12 weeks of history for percentile
            hist = atr_pct_1w[max(0, i-11):i+1]  # Last 12 weeks including current
            valid_hist = hist[~np.isnan(hist)]
            if len(valid_hist) >= 5:
                vol_threshold_1w[i] = np.percentile(valid_hist, 70)  # Top 30%
            else:
                vol_threshold_1w[i] = np.nan
        else:
            vol_threshold_1w[i] = np.nan
    
    # High volatility week filter
    high_vol_week = np.full(len(df_1w), False)
    for i in range(len(df_1w)):
        if not np.isnan(atr_pct_1w[i]) and not np.isnan(vol_threshold_1w[i]):
            high_vol_week[i] = atr_pct_1w[i] > vol_threshold_1w[i]
        else:
            high_vol_week[i] = False
    
    # Align weekly indicators to 6h timeframe
    high_vol_week_6h = align_htf_to_ltf(prices, df_1w, high_vol_week.astype(float))
    
    # Calculate 6-hour RSI (6-period) for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    if n >= 6:
        avg_gain[5] = np.mean(gain[:6])
        avg_loss[5] = np.mean(loss[:6])
        for i in range(6, n):
            avg_gain[i] = (avg_gain[i-1] * 5 + gain[i]) / 6
            avg_loss[i] = (avg_loss[i-1] * 5 + loss[i]) / 6
    
    rsi_6h = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(avg_loss[i]) and avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi_6h[i] = 100 - (100 / (1 + rs))
        elif not np.isnan(avg_gain[i]) and avg_loss[i] == 0:
            rsi_6h[i] = 100.0
    
    # Calculate 6-hour Bollinger Bands (20-period, 2 std)
    sma_20 = np.full(n, np.nan)
    std_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            sma_20[i] = np.mean(clone if 'clone' in locals() else close[i-19:i+1])
            std_20[i] = np.std(close[i-19:i+1])
    
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(sma_20[i]) and not np.isnan(std_20[i]):
            upper_band[i] = sma_20[i] + 2 * std_20[i]
            lower_band[i] = sma_20[i] - 2 * std_20[i]
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_6h[i]) or
            np.isnan(upper_band[i]) or
            np.isnan(lower_band[i]) or
            np.isnan(high_vol_week_6h[i])):
            signals[i] = 0.0
            continue
        
        # Only trade during high volatility weeks (top 30% volatility)
        if high_vol_week_6h[i] < 0.5:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) AND price touches or goes below lower Bollinger Band
            if rsi_6h[i] < 30 and close[i] <= lower_band[i]:
                position = 1
                signals[i] = position_size
            # Short: RSI > 70 (overbought) AND price touches or goes above upper Bollinger Band
            elif rsi_6h[i] > 70 and close[i] >= upper_band[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: RSI > 50 (mean reversion complete) OR price reaches middle (SMA)
            if rsi_6h[i] > 50 or close[i] >= sma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: RSI < 50 (mean reversion complete) OR price reaches middle (SMA)
            if rsi_6h[i] < 50 or close[i] <= sma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_HighVolWeek_BollingerRSI_MeanReversion"
timeframe = "6h"
leverage = 1.0