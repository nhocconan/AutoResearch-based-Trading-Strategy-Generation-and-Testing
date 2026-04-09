#!/usr/bin/env python3
# 1d_weekly_camarilla_pivot_volume_v2
# Hypothesis: Daily Camarilla pivot levels with weekly trend filter (price vs weekly EMA20) and volume confirmation.
# Works in bull/bear: Weekly EMA20 defines trend; Camarilla H3/L3 levels provide mean-reversion entries in range, breakouts in trend; volume avoids false signals.
# Target: 7-25 trades/year (30-100 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_camarilla_pivot_volume_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly HTF data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA20 for trend
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily data for Camarilla pivots
    # Need previous day's OHLC for today's Camarilla levels
    # We'll calculate Camarilla for each day using prior day's data
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    high_shift[0] = np.nan
    low_shift[0] = np.nan
    close_shift[0] = np.nan
    
    # Camarilla levels: based on previous day's range
    # H4 = close + 1.5*(high-low), H3 = close + 1.0*(high-low), etc.
    # L4 = close - 1.5*(high-low), L3 = close - 1.0*(high-low), etc.
    # We use H3/L3 for mean reversion, H4/L4 for breakouts
    rng = high_shift - low_shift
    H3 = close_shift + 1.0 * rng
    L3 = close_shift - 1.0 * rng
    H4 = close_shift + 1.5 * rng
    L4 = close_shift - 1.5 * rng
    
    # Volume confirmation: current volume > 1.5x 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup for Camarilla calculation
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(H3[i]) or np.isnan(L3[i]) or 
            np.isnan(H4[i]) or np.isnan(L4[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below L3 (mean reversion) OR below L4 (stop) OR weekly trend turns bearish
            if close[i] < L3[i] or close[i] < L4[i] or close[i] < ema_20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above H3 (mean reversion) OR above H4 (stop) OR weekly trend turns bullish
            if close[i] > H3[i] or close[i] > H4[i] or close[i] > ema_20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Determine market regime based on weekly trend
                weekly_bullish = close[i] > ema_20_1w_aligned[i]
                weekly_bearish = close[i] < ema_20_1w_aligned[i]
                
                if weekly_bullish:
                    # In bull trend: look for breakouts above H4 or pullbacks to L3
                    if close[i] > H4[i]:
                        position = 1
                        signals[i] = 0.25
                    elif close[i] < L3[i] and close[i] > L4[i]:  # Pullback to L3 but above L4
                        position = 1
                        signals[i] = 0.25
                elif weekly_bearish:
                    # In bear trend: look for breakdowns below L4 or pullbacks to H3
                    if close[i] < L4[i]:
                        position = -1
                        signals[i] = -0.25
                    elif close[i] > H3[i] and close[i] < H4[i]:  # Pullback to H3 but below H4
                        position = -1
                        signals[i] = -0.25
                else:
                    # Neutral/ranging: mean reversion at H3/L3
                    if close[i] < L3[i]:
                        position = 1
                        signals[i] = 0.25
                    elif close[i] > H3[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals