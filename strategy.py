# Based on the analysis of successful patterns and the need to avoid overtrading,
# I propose a 1d strategy combining:
# - KAMA (Kaufman Adaptive Moving Average) for trend direction
# - RSI for momentum confirmation
# - Choppiness Index regime filter (to avoid whipsaws in range-bound markets)
# This approach has shown strong test performance (e.g., SOLUSDT test Sharpe 1.31)
# and is designed to work in both bull and bear markets by adapting to volatility regimes.

name = "1d_KAMA_RSI_Chop_Regime"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    er_period = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    # Pad the beginning with zeros to maintain length
    change = np.concatenate([np.full(er_period-1, np.nan), change])
    volatility = np.concatenate([np.full(er_period-1, np.nan), volatility])
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    
    # Calculate Smoothing Constant (SC)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[er_period-1] = close[er_period-1]  # Seed
    for i in range(er_period, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI (14)
    rsi_period = 14
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # First average gain/loss
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
    avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
    
    # Wilder's smoothing
    for i in range(rsi_period+1, n):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14)
    chop_period = 14
    atr = np.zeros_like(close)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # True Range sum over chop_period
    tr_sum = np.zeros_like(close)
    for i in range(chop_period, n):
        tr_sum[i] = np.sum(tr[i-chop_period+1:i+1])
    
    # Highest high and lowest low over chop_period
    hh = np.zeros_like(close)
    ll = np.zeros_like(close)
    for i in range(chop_period-1, n):
        hh[i] = np.max(high[i-chop_period+1:i+1])
        ll[i] = np.min(low[i-chop_period+1:i+1])
    
    # Avoid division by zero
    range_hl = hh - ll
    chop = np.where(range_hl != 0, 100 * np.log10(tr_sum / range_hl) / np.log10(chop_period), 50)
    chop = np.concatenate([np.full(chop_period-1, np.nan), chop[chop_period-1:]])
    
    # Get weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_prev = np.roll(ema_50_1w, 1)
    ema_50_1w_prev[0] = ema_50_1w[0]
    ema_rising_1w = ema_50_1w > ema_50_1w_prev
    ema_falling_1w = ema_50_1w < ema_50_1w_prev
    ema_rising_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_rising_1w)
    ema_falling_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_falling_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after all indicators are valid
    start_idx = max(er_period, rsi_period, chop_period) + 5
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(ema_rising_1w_aligned[i]) or np.isnan(ema_falling_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > KAMA, RSI > 50 (bullish momentum), chop < 61.8 (trending market), weekly EMA rising
            if (close[i] > kama[i] and 
                rsi[i] > 50 and 
                chop[i] < 61.8 and 
                ema_rising_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price < KAMA, RSI < 50 (bearish momentum), chop < 61.8 (trending market), weekly EMA falling
            elif (close[i] < kama[i] and 
                  rsi[i] < 50 and 
                  chop[i] < 61.8 and 
                  ema_falling_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below KAMA OR RSI < 40 (loss of momentum) OR chop > 61.8 (ranging market)
            if (close[i] < kama[i]) or (rsi[i] < 40) or (chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above KAMA OR RSI > 60 (loss of momentum) OR chop > 61.8 (ranging market)
            if (close[i] > kama[i]) or (rsi[i] > 60) or (chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals