#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend + RSI(14) mean reversion + choppiness regime filter
# - KAMA adapts to market efficiency, trending in strong moves, flat in chop
# - RSI(14) < 30 for long, > 70 for short in ranging markets (CHOP > 61.8)
# - Choppiness index > 61.8 = ranging market (mean reversion regime)
# - Choppiness index < 38.2 = trending market (follow KAMA direction)
# - ATR(14) trailing stop (2.5x) for risk management
# - Position size: 0.25 discrete to minimize fee churn
# - Target: 15-25 trades/year (60-100 total over 4 years) to avoid fee drag
# - Works in bull/bear: regime filter adapts logic, KAMA avoids whipsaw, RSI captures reversals

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w EMA for trend filter (21-period)
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Pre-calculate indicators for primary timeframe (1d)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # KAMA (adaptive moving average)
    # Efficiency ratio: |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.subtract(close[10:], close[:-10]))  # |close - close[10]|
    volatility = np.sum(np.abs(np.subtract(close[1:], close[:-1])), axis=0)  # sum |close - close[1]|
    # Pad arrays for alignment
    change_padded = np.concatenate([np.full(10, np.nan), change])
    volatility_padded = np.concatenate([np.full(1, np.nan), volatility, np.full(9, np.nan)])
    # Calculate ER with proper alignment
    er = np.full_like(close, np.nan)
    valid_idx = ~(np.isnan(change_padded) | np.isnan(volatility_padded))
    er[valid_idx] = change_padded[valid_idx] / volatility_padded[valid_idx]
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum.reduce([tr1, tr2, tr3])
    # Sum of True Range over 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Choppiness Index
    chop = np.where(
        (atr_sum > 0) & (hh - ll > 0),
        100 * np.log10(atr_sum / (hh - ll)) / np.log10(14),
        50  # neutral when undefined
    )
    
    # ATR(14) for trailing stop
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0  # for trailing stop
    lowest_since_entry = 0.0   # for trailing stop
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(atr[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime filters
        chop_high = chop[i] > 61.8  # ranging market (mean reversion)
        chop_low = chop[i] < 38.2   # trending market
        
        if position == 0:  # Flat - look for new entries
            # Long conditions
            long_condition = False
            if chop_high:  # ranging market - mean reversion
                long_condition = rsi[i] < 30 and close[i] > kama[i]
            elif chop_low:  # trending market - follow trend
                long_condition = close[i] > kama[i] and close[i] > ema_21_1w_aligned[i]
            
            # Short conditions
            short_condition = False
            if chop_high:  # ranging market - mean reversion
                short_condition = rsi[i] > 70 and close[i] < kama[i]
            elif chop_low:  # trending market - follow trend
                short_condition = close[i] < kama[i] and close[i] < ema_21_1w_aligned[i]
            
            if long_condition:
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                highest_since_entry = prices['high'].iloc[i]
                signals[i] = 0.25
            elif short_condition:
                position = -1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                lowest_since_entry = prices['low'].iloc[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or trailing stop
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
                # Exit conditions: reverse signal OR ATR trailing stop
                exit_long = (
                    (chop_high and rsi[i] > 50) or  # RSI mean reversion exit in range
                    (chop_low and close[i] < kama[i]) or  # trend break
                    (close[i] < ema_21_1w_aligned[i])  # weekly trend break
                )
                trailing_stop = prices['close'].iloc[i] < highest_since_entry - 2.5 * atr[i]
                exit_condition = exit_long or trailing_stop
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # Exit conditions: reverse signal OR ATR trailing stop
                exit_short = (
                    (chop_high and rsi[i] < 50) or  # RSI mean reversion exit in range
                    (chop_low and close[i] > kama[i]) or  # trend break
                    (close[i] > ema_21_1w_aligned[i])  # weekly trend break
                )
                trailing_stop = prices['close'].iloc[i] > lowest_since_entry + 2.5 * atr[i]
                exit_condition = exit_short or trailing_stop
            
            if exit_condition:
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals