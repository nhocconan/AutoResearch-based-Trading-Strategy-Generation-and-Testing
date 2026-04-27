# 1d_WeeklyCandlestickPattern_SMTrend
# Hypothesis: Daily timeframe uses weekly candlestick patterns (engulfing, hammer) for reversal signals,
# filtered by weekly EMA20 trend direction to trade with higher timeframe momentum.
# Works in bull/bear by only taking long patterns in uptrend, short patterns in downtrend.
# Low frequency: expects 10-25 trades/year by requiring weekly pattern + trend alignment.
# Uses engulfing/bullish/bearish patterns with body size > 50% of candle range for reliability.
# Includes volatility-based stop via EMA distance exit.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get weekly data for pattern and trend
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    wo = df_weekly['open'].values
    wh = df_weekly['high'].values
    wl = df_weekly['low'].values
    wc = df_weekly['close'].values
    n_weekly = len(wc)
    
    # Weekly EMA20 for trend filter
    ema20_weekly = np.full(n_weekly, np.nan)
    if n_weekly >= 20:
        ema20_weekly[19] = np.mean(wc[:20])
        for i in range(20, n_weekly):
            ema20_weekly[i] = (wc[i] * 2/21) + (ema20_weekly[i-1] * 19/21)
    
    # Weekly candlestick pattern detection
    body = np.abs(wc - wo)
    rng = wh - wl
    body_ratio = np.where(rng > 0, body / rng, 0)
    
    # Bullish engulfing: current green engulfs previous red
    bull_engulf = np.zeros(n_weekly, dtype=bool)
    for i in range(1, n_weekly):
        if (wc[i] > wo[i] and  # current green
            wc[i-1] < wo[i-1] and  # previous red
            wc[i] >= wo[i-1] and  # current close >= prev open
            wo[i] <= wc[i-1] and  # current open <= prev close
            body_ratio[i] > 0.5):  # meaningful body
            bull_engulf[i] = True
    
    # Bearish engulfing: current red engulfs previous green
    bear_engulf = np.zeros(n_weekly, dtype=bool)
    for i in range(1, n_weekly):
        if (wc[i] < wo[i] and  # current red
            wc[i-1] > wo[i-1] and  # previous green
            wc[i] <= wo[i-1] and  # current close <= prev open
            wo[i] >= wc[i-1] and  # current open >= prev close
            body_ratio[i] > 0.5):  # meaningful body
            bear_engulf[i] = True
    
    # Hammer: small top body, long lower shadow
    upper_shadow = wh - np.maximum(wc, wo)
    lower_shadow = np.minimum(wc, wo) - wl
    hammer = np.zeros(n_weekly, dtype=bool)
    for i in range(n_weekly):
        if (body[i] > 0 and  # has body
            lower_shadow[i] > 2 * body[i] and  # long lower shadow
            upper_shadow[i] < 0.5 * body[i] and  # small upper shadow
            body_ratio[i] > 0.3):  # decent body
            hammer[i] = True
    
    # Inverted hammer: small bottom body, long upper shadow
    inv_hammer = np.zeros(n_weekly, dtype=bool)
    for i in range(n_weekly):
        if (body[i] > 0 and  # has body
            upper_shadow[i] > 2 * body[i] and  # long upper shadow
            lower_shadow[i] < 0.5 * body[i] and  # small lower shadow
            body_ratio[i] > 0.3):  # decent body
            inv_hammer[i] = True
    
    # Align weekly signals to daily
    ema20_aligned = align_htf_to_ltf(prices, df_weekly, ema20_weekly)
    bull_engulf_aligned = align_htf_to_ltf(prices, df_weekly, bull_engulf.astype(float))
    bear_engulf_aligned = align_htf_to_ltf(prices, df_weekly, bear_engulf.astype(float))
    hammer_aligned = align_htf_to_ltf(prices, df_weekly, hammer.astype(float))
    inv_hammer_aligned = align_htf_to_ltf(prices, df_weekly, inv_hammer.astype(float))
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position
    
    # Warmup: need weekly EMA20 + pattern lookback
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(ema20_aligned[i]) or 
            np.isnan(bull_engulf_aligned[i]) or
            np.isnan(bear_engulf_aligned[i]) or
            np.isnan(hammer_aligned[i]) or
            np.isnan(inv_hammer_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema20 = ema20_aligned[i]
        
        # Determine weekly trend from aligned EMA
        # Need previous weekly bar's EMA to determine trend direction
        # Use current price vs EMA for simplicity
        above_ema = price > ema20
        below_ema = price < ema20
        
        if position == 0:
            # Long signals: bullish patterns in uptrend (price > EMA20)
            if above_ema and (bull_engulf_aligned[i] > 0.5 or hammer_aligned[i] > 0.5):
                signals[i] = size
                position = 1
            # Short signals: bearish patterns in downtrend (price < EMA20)
            elif below_ema and (bear_engulf_aligned[i] > 0.5 or inv_hammer_aligned[i] > 0.5):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below EMA20 or opposite pattern appears
            if below_ema or (bear_engulf_aligned[i] > 0.5 or inv_hammer_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above EMA20 or opposite pattern appears
            if above_ema or (bull_engulf_aligned[i] > 0.5 or hammer_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyCandlestickPattern_SMTrend"
timeframe = "1d"
leverage = 1.0