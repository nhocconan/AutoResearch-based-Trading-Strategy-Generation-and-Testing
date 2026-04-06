# 1d Weekly Pivot + Volume Confirmation + ATR Stop
# Hypothesis: Weekly pivot levels from weekly timeframe combined with daily trend filter and volume confirmation.
# Works in bull markets (breakouts of R4/S4 with trend) and bear markets (breakdowns with trend).
# Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weeklypivot_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # 1w data for weekly pivot levels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot from previous week's data
    pivot_1w = np.full_like(close_1w, np.nan)
    r1_1w = np.full_like(close_1w, np.nan)
    s1_1w = np.full_like(close_1w, np.nan)
    r2_1w = np.full_like(close_1w, np.nan)
    s2_1w = np.full_like(close_1w, np.nan)
    r3_1w = np.full_like(close_1w, np.nan)
    s3_1w = np.full_like(close_1w, np.nan)
    r4_1w = np.full_like(close_1w, np.nan)
    s4_1w = np.full_like(close_1w, np.nan)
    
    # Calculate pivots for each weekly bar (using previous week's OHLC)
    for i in range(1, len(close_1w)):
        # Use previous week's OHLC to calculate current week's pivot
        ph = high_1w[i-1]
        pl = low_1w[i-1]
        pc = close_1w[i-1]
        
        if not (np.isnan(ph) or np.isnan(pl) or np.isnan(pc)):
            p = (ph + pl + pc) / 3.0
            pivot_1w[i] = p
            r1_1w[i] = 2*p - pl
            s1_1w[i] = 2*p - ph
            r2_1w[i] = p + (ph - pl)
            s2_1w[i] = p - (ph - pl)
            r3_1w[i] = ph + 2*(p - pl)
            s3_1w[i] = pl - 2*(ph - p)
            r4_1w[i] = 3*p - 2*pl
            s4_1w[i] = 3*ph - 2*pl
    
    # Align weekly pivot levels to daily timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # 1d EMA50 for trend bias
    ema_1d = np.full(n, np.nan)
    if n >= 50:
        ema_1d[49] = np.mean(close[:50])
        for i in range(50, n):
            ema_1d[i] = (close[i] * 2 + ema_1d[i-1] * 18) / 20
    
    # Trend bias: above EMA = bullish, below = bearish
    trend_bias = np.where(close > ema_1d, 1, -1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 50  # Need enough data for EMA and pivots
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(trend_bias[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[max(0, i-20):i])
        volume_filter = volume[i] > vol_ma * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price drops below S3 (mean reversion) OR against daily trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < s3_aligned[i] or
                trend_bias[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price rises above R3 (mean reversion) OR against daily trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > r3_aligned[i] or
                trend_bias[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 20 bars flat
            if bars_since_entry >= 20:
                # Breakout entries: R4/S4 with trend
                bull_breakout = close[i] > r4_aligned[i]
                bear_breakout = close[i] < s4_aligned[i]
                
                # Mean reversion entries: R3/S3 counter-trend (fade)
                # Only when near pivot as proxy for ranging markets
                pivot_range = r1_aligned[i] - s1_aligned[i]
                near_pivot = abs(close[i] - pivot_aligned[i]) < pivot_range * 0.5
                
                # Long: breakout with trend OR mean reversion at S3 with volume
                if (bull_breakout and trend_bias[i] == 1 and volume_filter) or \
                   (close[i] > s3_aligned[i] and close[i] < pivot_aligned[i] and 
                    near_pivot and volume_filter and trend_bias[i] == -1):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown with trend OR mean reversion at R3 with volume
                elif (bear_breakout and trend_bias[i] == -1 and volume_filter) or \
                     (close[i] < r3_aligned[i] and close[i] > pivot_aligned[i] and 
                      near_pivot and volume_filter and trend_bias[i] == 1):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals