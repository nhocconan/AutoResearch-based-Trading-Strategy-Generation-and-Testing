#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly RSI filter and volatility-adjusted position sizing.
# Uses Donchian channel breakouts for trend capture, weekly RSI(14) to avoid extremes,
# and ATR-based volatility scaling to reduce size in high volatility regimes.
# Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drag.
# Works in bull markets via trend following and in bear markets via mean reversion at RSI extremes.

name = "daily_donchian20_weekly_rsi_vol_scaled_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Weekly data for RSI filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly RSI(14)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # ATR(14) for volatility scaling
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma = pd.Series(atr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(rsi_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(atr_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime: scale position size inversely with volatility
        vol_ratio = atr[i] / atr_ma[i] if atr_ma[i] > 0 else 1.0
        vol_scale = np.clip(1.0 / vol_ratio, 0.5, 1.5)  # scale between 0.5 and 1.5
        base_size = 0.25
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i-1]  # break above previous period's high
        breakout_down = close[i] < lowest_low[i-1]  # break below previous period's low
        
        # Extreme RSI conditions for mean reversion (avoid trading against strong momentum)
        rsi_not_extreme = (rsi_1w_aligned[i] >= 30) and (rsi_1w_aligned[i] <= 70)
        oversold = rsi_1w_aligned[i] < 30
        overbought = rsi_1w_aligned[i] > 70
        
        # Long conditions: bullish breakout OR oversold mean reversion (only if not in extreme overbought)
        if breakout_up or (oversold and rsi_not_extreme):
            signals[i] = base_size * vol_scale
        # Short conditions: bearish breakout OR overbought mean reversion (only if not in extreme oversold)
        elif breakout_down or (overbought and rsi_not_extreme):
            signals[i] = -base_size * vol_scale
        else:
            signals[i] = 0.0
    
    return signals