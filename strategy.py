#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout + 4h trend filter + session filter
# - Primary signal: 1h price breaks above Camarilla H3 (long) or below L3 (short)
# - Trend filter: 4h close above/below 20-period EMA for directional bias
# - Session filter: only trade 08:00-20:00 UTC to avoid low-volume Asian session
# - Position size: 0.20 discrete level to minimize fee churn
# - Target: 15-37 trades/year (60-150 total over 4 years) per 1h strategy guidelines
# - Works in bull/bear: Camarilla pivots adapt to volatility, 4h EMA filter avoids counter-trend whipsaws

name = "1h_4h_camarilla_breakout_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Pre-compute 4h EMA20 for trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Pre-compute 1h Camarilla levels (using previous day's OHLC)
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    # Calculate daily pivot points (using prior day's OHLC)
    # Shift by 1 to use previous day's data (no look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # First bar has no prior day
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    h3 = pivot + (range_hl * 1.1 / 4.0)  # H3 = pivot + 1.1*(H-L)/4
    l3 = pivot - (range_hl * 1.1 / 4.0)  # L3 = pivot - 1.1*(H-L)/4
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20_4h_aligned[i]) or
            np.isnan(h3[i]) or
            np.isnan(l3[i]) or
            np.isnan(in_session[i])):
            signals[i] = 0.0
            continue
        
        if not in_session[i]:
            # Outside session: flatten position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below L3 (reversion to mean) OR 4h trend turns bearish
            if prices['close'].iloc[i] < l3[i] or prices['close'].iloc[i] < ema_20_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price crosses above H3 (reversion to mean) OR 4h trend turns bullish
            if prices['close'].iloc[i] > h3[i] or prices['close'].iloc[i] > ema_20_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Camarilla breakouts with 4h trend alignment
            # Long: price breaks above H3 AND 4h close above EMA20 (bullish trend)
            if (prices['close'].iloc[i] > h3[i] and 
                prices['close'].iloc[i] > ema_20_4h_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short: price breaks below L3 AND 4h close below EMA20 (bearish trend)
            elif (prices['close'].iloc[i] < l3[i] and 
                  prices['close'].iloc[i] < ema_20_4h_aligned[i]):
                position = -1
                signals[i] = -0.20
    
    return signals