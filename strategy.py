#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction + RSI(14) + chop regime filter (CHOP > 61.8 = range) for mean reversion in choppy markets.
# Uses 1w EMA50 for trend filter to avoid counter-trend trades in strong trends.
# Long when KAMA upward AND RSI < 30 AND CHOP > 61.8 (oversold in range).
# Short when KAMA downward AND RSI > 70 AND CHOP > 61.8 (overbought in range).
# Exit on opposite RSI extreme (RSI > 70 for long, RSI < 30 for short) or trend change (price crosses 1w EMA50).
# Uses discrete sizing 0.25 to manage drawdown and reduce fee churn.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.

name = "1d_KAMA_RSI_Chop_Regime_MeanReversion_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_time = prices['open_time']
    
    # Pre-compute session hours (optional for 1d, but kept for consistency)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 1d data (primary) and 1w data (HTF trend filter) ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 30 or len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d KAMA calculation (ER = 10, fast = 2, slow = 30)
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder, will compute correctly below
    # Correct volatility calculation: sum of absolute changes over ER period
    er_period = 10
    volatility = pd.Series(close_1d).rolling(window=er_period, min_periods=er_period).apply(
        lambda x: np.sum(np.abs(np.diff(x))), raw=True
    ).values
    # Avoid division by zero
    change_vs_volatility = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    er = change_vs_volatility
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # smoothing constant
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # 1d RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d Choppiness Index (CHOP) - using 14-period
    # CHOP = 100 * log10(sum(ATR(1)) / (n * log10(highest_high - lowest_low))) / log10(n)
    atr_period = 14
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.abs(low[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])  # align with close_1d
    atr = pd.Series(tr1).ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values
    sum_atr = pd.Series(atr).rolling(window=atr_period, min_periods=atr_period).sum().values
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    # Avoid log of zero or negative
    range_hl = highest_high - lowest_low
    log_sum_atr = np.log10(np.where(sum_atr > 0, sum_atr, np.nan))
    log_range = np.log10(np.where(range_hl > 0, range_hl, np.nan))
    log_n = np.log10(atr_period)
    chop = 100 * (log_sum_atr / (atr_period * log_range)) / log_n
    # Handle edge cases
    chop = np.where((log_range > 0) & (log_sum_atr > 0), chop, 50.0)  # default to neutral
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Align 1d indicators to lower timeframe (1d bars are already LTF in this case)
    # Since timeframe is 1d, we can use the 1d arrays directly but need to align to prices index
    # Map 1d data to 1d prices (they should match, but we use align for safety)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Trend condition: price above/below 1w EMA50
    price_above_1w_ema = close > ema_50_1w_aligned
    price_below_1w_ema = close < ema_50_1w_aligned
    
    # KAMA direction: upward if today's KAMA > yesterday's KAMA
    kama_up = kama_aligned > np.concatenate([[kama_aligned[0]], kama_aligned[:-1]])
    kama_down = kama_aligned < np.concatenate([[kama_aligned[0]], kama_aligned[:-1]])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Optional session filter (08-20 UTC) - can be relaxed for 1d
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i]) or np.isnan(ema_50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Mean reversion conditions in choppy market (CHOP > 61.8 = range)
        in_range = chop_aligned[i] > 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long: KAMA up AND RSI < 30 (oversold) AND in range
            if (kama_up[i] and 
                rsi_aligned[i] < 30 and 
                in_range):
                signals[i] = 0.25
                position = 1
            # Short: KAMA down AND RSI > 70 (overbought) AND in range
            elif (kama_down[i] and 
                  rsi_aligned[i] > 70 and 
                  in_range):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: RSI > 70 (overbought) OR price < 1w EMA50 (trend change to down)
            if (rsi_aligned[i] > 70 or 
                not price_above_1w_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: RSI < 30 (oversold) OR price > 1w EMA50 (trend change to up)
            if (rsi_aligned[i] < 30 or 
                not price_below_1w_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals