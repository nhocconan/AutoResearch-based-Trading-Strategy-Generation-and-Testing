#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion with 4h trend filter and 1d regime filter
# - Uses 4h EMA(50) for trend direction (long when price > EMA50, short when price < EMA50)
# - Uses 1d chop regime filter (choppiness > 61.8 = range = mean revert)
# - 1h entry: RSI(14) < 30 for long, RSI(14) > 70 for short (only in ranging markets)
# - Fixed position size 0.20 to control drawdown
# - Target: 15-35 trades/year on 1h timeframe (60-140 total over 4 years)
# - Works in bull markets via 4h trend filter, works in bear/ranging via chop regime + mean reversion

name = "1h_4h_1d_chop_regime_meanrev_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Pre-compute session filter (08-20 UTC) - prices.index is DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend direction
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d Choppiness Index (CHOP) for regime detection
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (HHV(high,14) - LLV(low,14))))
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    hh_14_1d = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    ll_14_1d = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    range_14_1d = hh_14_1d - ll_14_1d
    
    # Avoid division by zero
    range_14_1d = np.where(range_14_1d == 0, 1e-10, range_14_1d)
    
    # Choppiness Index calculation
    log_sum_atr = np.log10(atr_14_1d)
    log_n = np.log10(14)
    chop_1d = 100 * (log_sum_atr / (log_n * np.log10(range_14_1d)))
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Pre-compute 1h RSI(14) for entry signals
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(chop_1d_aligned[i]) or
            np.isnan(rsi[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in ranging markets (CHOP > 61.8)
        if chop_1d_aligned[i] <= 61.8:
            signals[i] = 0.0
            continue
        
        # Trend filter from 4h EMA(50)
        price_above_ema = close[i] > ema_50_4h_aligned[i]
        price_below_ema = close[i] < ema_50_4h_aligned[i]
        
        # Mean reversion entry signals from 1h RSI
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Long signal: ranging market + price above 4h EMA (uptrend bias) + RSI oversold
        if rsi_oversold and price_above_ema:
            signals[i] = 0.20
        # Short signal: ranging market + price below 4h EMA (downtrend bias) + RSI overbought
        elif rsi_overbought and price_below_ema:
            signals[i] = -0.20
    
    return signals