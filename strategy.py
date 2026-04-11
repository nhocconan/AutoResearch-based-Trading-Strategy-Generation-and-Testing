#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_orb_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-calculate session mask (UTC 08-20)
    session_mask = (prices.index.hour >= 8) & (prices.index.hour < 20)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily 50-period EMA for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Daily ATR (14-period) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = np.roll(close_1d, 1)
    close_1d_prev[0] = close_1d[0]
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d_prev)
    tr3 = np.abs(low_1d - close_1d_prev)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Opening range: first 2 hours of session (08:00-10:00 UTC)
    # We'll calculate this daily
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    or_high = np.nan
    or_low = np.nan
    or_valid = False
    
    # Track last date to reset OR
    last_date = None
    
    for i in range(50, n):  # Start after warmup for daily indicators
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current timestamp and date
        current_time = prices.index[i]
        current_date = current_time.date()
        
        # Reset opening range at start of new day
        if last_date != current_date:
            last_date = current_date
            or_high = np.nan
            or_low = np.nan
            or_valid = False
        
        # Update opening range (08:00-10:00 UTC)
        hour = current_time.hour
        minute = current_time.minute
        
        if 8 <= hour < 10:  # During OR period
            if np.isnan(or_high):
                or_high = high[i]
                or_low = low[i]
            else:
                or_high = max(or_high, high[i])
                or_low = min(or_low, low[i])
        elif hour >= 10 and not or_valid:  # OR period just ended
            or_valid = True
        
        # Skip if OR not yet formed or outside session
        if not or_valid or not session_mask[i]:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        ema_50 = ema_50_1d_aligned[i]
        atr_14 = atr_14_1d_aligned[i]
        
        # Trend filter: price above/below daily EMA50
        uptrend = price_close > ema_50
        downtrend = price_close < ema_50
        
        # Volatility filter: avoid low volatility periods
        vol_filter = atr_14 > 0.01 * ema_50  # ATR > 1% of price
        
        # Breakout signals
        long_breakout = price_high > or_high
        short_breakout = price_low < or_low
        
        # Entry conditions
        long_signal = long_breakout and uptrend and vol_filter
        short_signal = short_breakout and downtrend and vol_filter
        
        # Exit conditions: time-based or mean reversion to OR midpoint
        if position == 1:
            # Exit if price returns to OR midpoint or reverses below OR low
            or_mid = (or_high + or_low) / 2.0
            exit_long = price_close < or_mid or price_low < or_low
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit if price returns to OR midpoint or reverses above OR high
            or_mid = (or_high + or_low) / 2.0
            exit_short = price_close > or_mid or price_high > or_high
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        elif long_signal:
            position = 1
            entry_price = price_close
            signals[i] = 0.20
        elif short_signal:
            position = -1
            entry_price = price_close
            signals[i] = -0.20
        else:
            # Maintain current position or flat
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Opening Range Breakout (ORB) with daily trend and volatility filters.
# - Uses 1h timeframe for precise entry timing during active session (08-20 UTC).
# - Calculates opening range (first 2 hours: 08:00-10:00 UTC) each day.
# - Enters long when price breaks above OR high in uptrend (price > daily EMA50).
# - Enters short when price breaks below OR low in downtrend (price < daily EMA50).
# - Volatility filter ensures sufficient ATR (>1% of price) to avoid choppy markets.
# - Exits when price returns to OR midpoint or shows reversal signals.
# - Designed for low trade frequency: ~1-2 breakouts per day max, yielding ~300-600 trades over 4 years.
# - Works in both bull and bear markets by trading breakouts in direction of higher timeframe trend.
# - Session filter avoids low-volume overnight periods.