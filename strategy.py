#!/usr/bin/env python3
"""
EXPERIMENT #14114 - 1h Strategy with 4h/1d Multi-Timeframe
Hypothesis: Use 4h trend (EMA50/200) and 1d volatility regime (ATR percentile) 
for directional bias, with 1h RSI pullback entries during London/NY session (08-20 UTC).
Designed to work in both bull/bear markets by filtering counter-trend trades.
Target: 60-150 trades over 4 years (15-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14114_1h_ema_rsi_session_v1"
timeframe = "1h"
leverage = 1.0

def calculate_rsi(close, period):
    """Calculate RSI with proper min_periods"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # === PRE-COMPUTE INDICATORS BEFORE LOOP ===
    
    # 1. 4h EMA trend filter (primary trend direction)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # 2. 1d ATR percentile for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    # Calculate percentile rank of current ATR vs 50-day lookback
    atr_percentile = pd.Series(atr_1d).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # 3. 1h RSI for entry timing
    close = prices['close'].values
    rsi = calculate_rsi(close, 14)
    
    # 4. Session filter (08-20 UTC)
    hours = prices.index.hour  # Already datetime64[ms], .hour works
    session_filter = (hours >= 8) & (hours <= 20)
    
    # 5. Volume filter (avoid low liquidity periods)
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (0.5 * vol_ma)  # At least 50% of average volume
    
    # === SIGNAL GENERATION LOOP ===
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start = max(200, 50, 20) + 1  # Warmup for longest indicator
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(ema200_4h_aligned[i]) or 
            np.isnan(atr_percentile_aligned[i]) or np.isnan(rsi[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check if we're in preferred volatility regime (not too choppy, not too volatile)
        vol_regime_ok = 0.3 <= atr_percentile_aligned[i] <= 0.8
        
        # Determine 4h trend bias
        uptrend_4h = ema50_4h_aligned[i] > ema200_4h_aligned[i]
        downtrend_4h = ema50_4h_aligned[i] < ema200_4h_aligned[i]
        
        # RSI conditions for pullback entries
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        # Exit conditions: trend reversal or extreme RSI
        exit_long = not uptrend_4h or rsi[i] > 75
        exit_short = not downtrend_4h or rsi[i] < 25
        
        # Manage existing positions
        if position == 1:  # Long position
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:  # No position - look for entry
            if (session_filter[i] and vol_filter[i] and vol_regime_ok):
                # Long: uptrend + RSI pullback from oversold
                if uptrend_4h and rsi_oversold and rsi[i] > rsi[i-1]:  # RSI turning up
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                # Short: downtrend + RSI pullback from overbought
                elif downtrend_4h and rsi_overbought and rsi[i] < rsi[i-1]:  # RSI turning down
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals