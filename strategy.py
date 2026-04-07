#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Daily KAMA + RSI with Chop Filter
# Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, providing a reliable trend filter.
# RSI identifies overbought/oversold conditions for mean reversion entries.
# Choppiness index determines market regime: high chop = range (mean revert), low chop = trend (follow trend).
# Works in both bull and bear markets: In choppy markets, mean reversion at RSI extremes; in trending markets, follow KAMA direction.
# Uses daily timeframe for trend/regime filters to reduce noise and overtrading.
# Target: 12-37 trades/year (50-150 over 4 years).

name = "12h_daily_kama_rsi_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for indicators
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    daily_close = df_daily['close'].values
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily close
    # ER (Efficiency Ratio) = |net change| / sum(|absolute changes|)
    change = np.abs(np.diff(daily_close))
    abs_change = np.abs(np.diff(daily_close))
    # Pad first element
    change = np.concatenate([[0], change])
    abs_change = np.concatenate([[0], abs_change])
    
    # Calculate ER over 10 periods
    er = np.zeros_like(daily_close)
    for i in range(10, len(daily_close)):
        net_change = np.abs(daily_close[i] - daily_close[i-10])
        total_change = np.sum(abs_change[i-9:i+1])
        if total_change > 0:
            er[i] = net_change / total_change
        else:
            er[i] = 0
    # Smooth ER
    sc = (er * 0.29 + 0.06) ** 2  # smoothing constant
    kama = np.zeros_like(daily_close)
    kama[0] = daily_close[0]
    for i in range(1, len(daily_close)):
        kama[i] = kama[i-1] + sc[i] * (daily_close[i] - kama[i-1])
    
    # Calculate RSI (14) on daily close
    delta = np.diff(daily_close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(daily_close)
    avg_loss = np.zeros_like(daily_close)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    for i in range(15, len(daily_close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    # Handle division by zero
    rsi = np.where(avg_loss == 0, 100, rsi)
    rsi = np.where(avg_gain == 0, 0, rsi)
    
    # Calculate Choppiness Index (14) on daily data
    # True Range
    tr1 = daily_high[1:] - daily_low[1:]
    tr2 = np.abs(daily_high[1:] - daily_close[:-1])
    tr3 = np.abs(daily_low[1:] - daily_close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[0], tr])  # align with index
    # Sum of TR over 14 periods
    tr_sum = np.zeros_like(daily_close)
    for i in range(14, len(tr)):
        tr_sum[i] = np.sum(tr[i-13:i+1])
    # Highest high and lowest low over 14 periods
    hh = np.zeros_like(daily_close)
    ll = np.zeros_like(daily_close)
    for i in range(14, len(daily_close)):
        hh[i] = np.max(daily_high[i-13:i+1])
        ll[i] = np.min(daily_low[i-13:i+1])
    # Chop calculation
    chop = np.zeros_like(daily_close)
    for i in range(14, len(daily_close)):
        if hh[i] - ll[i] > 0:
            chop[i] = 100 * np.log10(tr_sum[i] / (hh[i] - ll[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    
    # Align indicators to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_daily, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_daily, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_daily, chop)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 70 or chop < 38 (strong trend) or volume drops
            if (rsi_aligned[i] > 70 or chop_aligned[i] < 38 or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: RSI < 30 or chop < 38 (strong trend) or volume drops
            if (rsi_aligned[i] < 30 or chop_aligned[i] < 38 or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Mean reversion in choppy market: chop > 60
            if chop_aligned[i] > 60:
                # Long: RSI < 30 and price > KAMA (bullish bias in range)
                if rsi_aligned[i] < 30 and close[i] > kama_aligned[i] and vol_filter[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: RSI > 70 and price < KAMA (bearish bias in range)
                elif rsi_aligned[i] > 70 and close[i] < kama_aligned[i] and vol_filter[i]:
                    position = -1
                    signals[i] = -0.25
            # Trend following in trending market: chop < 40
            elif chop_aligned[i] < 40:
                # Long: price > KAMA and RSI > 50
                if close[i] > kama_aligned[i] and rsi_aligned[i] > 50 and vol_filter[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price < KAMA and RSI < 50
                elif close[i] < kama_aligned[i] and rsi_aligned[i] < 50 and vol_filter[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals