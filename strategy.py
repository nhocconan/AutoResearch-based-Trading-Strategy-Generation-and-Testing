#!/usr/bin/env python3
# 1h_Regime_Adaptive_4hTrend_1dMomentum
# Hypothesis: In 1h timeframe, use 4h trend direction (EMA20) and 1d momentum (RSI14) to filter entries.
# Long when: price > 4h EMA20 AND 1d RSI > 50 AND 1h RSI < 30 (pullback in uptrend)
# Short when: price < 4h EMA20 AND 1d RSI < 50 AND 1h RSI > 70 (bounce in downtrend)
# Adds session filter (08-20 UTC) to avoid low-liquidity hours.
# Position size fixed at 0.20 to limit drawdown. Target: 20-40 trades/year.
# Designed to work in both bull (trend following on pullbacks) and bear (mean reversion in downtrend) markets.

name = "1h_Regime_Adaptive_4hTrend_1dMomentum"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === 4h EMA20 Trend Filter ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # === 1d RSI14 for Regime ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi14_1d = 100 - (100 / (1 + rs))
    rsi14_1d = rsi14_1d.replace([np.inf, -np.inf], 100).fillna(50).values
    rsi14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi14_1d)
    
    # === 1h RSI14 for Entry Timing ===
    delta_1h = pd.Series(close).diff()
    gain_1h = delta_1h.clip(lower=0)
    loss_1h = -delta_1h.clip(upper=0)
    avg_gain_1h = gain_1h.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_1h = loss_1h.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_1h = avg_gain_1h / avg_loss_1h
    rsi14_1h = 100 - (100 / (1 + rs_1h))
    rsi14_1h = rsi14_1h.replace([np.inf, -np.inf], 100).fillna(50).values
    
    # === Signal Parameters ===
    position_size = 0.20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for RSI)
    start_idx = 30  # covers 14+14 for safety
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(rsi14_1d_aligned[i]) or 
            np.isnan(rsi14_1h[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Uptrend (price > 4h EMA20) + bullish regime (1d RSI > 50) + oversold entry (1h RSI < 30)
            if (close[i] > ema20_4h_aligned[i] and 
                rsi14_1d_aligned[i] > 50 and 
                rsi14_1h[i] < 30):
                signals[i] = position_size
                position = 1
            # Short: Downtrend (price < 4h EMA20) + bearish regime (1d RSI < 50) + overbought entry (1h RSI > 70)
            elif (close[i] < ema20_4h_aligned[i] and 
                  rsi14_1d_aligned[i] < 50 and 
                  rsi14_1h[i] > 70):
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions: reverse signal or RSI mean reversion
            if position == 1:
                # Exit: 1h RSI crosses above 70 (overbought) or trend breaks
                if (rsi14_1h[i] >= 70 or close[i] < ema20_4h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: 1h RSI crosses below 30 (oversold) or trend breaks
                if (rsi14_1h[i] <= 30 or close[i] > ema20_4h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals