#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and session filter
# - Long when price breaks above H3 pivot level AND 4h close > 4h EMA50 (bullish trend)
# - Short when price breaks below L3 pivot level AND 4h close < 4h EMA50 (bearish trend)
# - Session filter: only trade 08:00-20:00 UTC to avoid low-volume periods
# - Exit: opposite pivot breakout or trend reversal
# - Position sizing: 0.20 discrete level to minimize fee drag
# - Target: 15-37 trades/year on 1h timeframe to stay within fee drag limits
# - Works in bull/bear: trend filter ensures we trade with higher timeframe momentum

name = "1h_4h_camarilla_pivot_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 1h Camarilla pivots (based on previous bar)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # H3 = Pivot + Range * 1.1/2
    # L3 = Pivot - Range * 1.1/2
    pivot = (high + low + close) / 3.0
    rng = high - low
    h3 = pivot + rng * 1.1 / 2.0
    l3 = pivot - rng * 1.1 / 2.0
    
    # Use previous bar's levels for breakout (no look-ahead)
    h3_prev = np.roll(h3, 1)
    l3_prev = np.roll(l3, 1)
    h3_prev[0] = np.nan
    l3_prev[0] = np.nan
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h close for trend comparison
    close_4h_aligned = align_htf_to_ltf(prices, df_4h, close_4h)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(60, n):  # Start after warmup for 4h EMA50
        # Skip if any required data is invalid
        if (np.isnan(h3_prev[i]) or np.isnan(l3_prev[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(close_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: 4h close vs 4h EMA50
        trend_bullish = close_4h_aligned[i] > ema_50_4h_aligned[i]
        trend_bearish = close_4h_aligned[i] < ema_50_4h_aligned[i]
        
        # Camarilla breakout signals
        breakout_up = close[i] > h3_prev[i]  # Break above H3
        breakout_down = close[i] < l3_prev[i]  # Break below L3
        
        # Exit conditions: opposite breakout or trend reversal
        exit_long = breakout_down or not trend_bullish
        exit_short = breakout_up or not trend_bearish
        
        if position == 0:  # Flat - look for entry
            if breakout_up and trend_bullish:
                position = 1
                signals[i] = 0.20
            elif breakout_down and trend_bearish:
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
    
    return signals