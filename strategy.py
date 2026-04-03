#!/usr/bin/env python3
"""
Experiment #014: 1h RSI(14) Extreme Reversion + 4h/1d Trend Filter + Session Filter (08-20 UTC)
HYPOTHESIS: In both bull and bear markets, extreme RSI readings (<30 or >70) on 1h often precede mean-reversion moves.
We filter these signals using 4h price > 200-EMA for bullish bias and 1d price < 200-EMA for bearish bias to align with higher timeframe trend.
Only trade during 08-20 UTC session to avoid low-liquidity periods. Fixed size 0.20 minimizes fee impact.
Target: 60-150 trades over 4 years (15-37/year) by requiring multiple confluence factors.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_extreme_4h_1d_ema_session_v1"
timeframe = "1h"
leverage = 1.0

def calculate_ema(close, period):
    """Exponential Moving Average"""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # === HTF: 4h EMA200 for bullish bias, 1d EMA200 for bearish bias (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    ema_4h_200 = calculate_ema(df_4h['close'].values, 200)
    ema_4h_200_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_200)
    
    df_1d = get_htf_data(prices, '1d')
    ema_1d_200 = calculate_ema(df_1d['close'].values, 200)
    ema_1d_200_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_200)
    
    # === 1h Indicators ===
    rsi_14 = calculate_rsi(close, period=14)
    
    # === Session filter: 08-20 UTC (pre-compute hours array) ===
    hours = pd.DatetimeIndex(open_time).hour
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Fixed position size (20% of capital) to minimize fee churn
    
    warmup = 200  # Ensure enough data for EMA200 and RSI
    
    for i in range(warmup, n):
        # Skip if any indicator is NaN
        if (np.isnan(rsi_14[i]) or np.isnan(ema_4h_200_aligned[i]) or 
            np.isnan(ema_1d_200_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # === Entry Conditions ===
        # Bullish: RSI < 30 (oversold) + price > 4h EMA200 (bullish bias on 4h)
        bullish_setup = (rsi_14[i] < 30) and (close[i] > ema_4h_200_aligned[i])
        
        # Bearish: RSI > 70 (overbought) + price < 1d EMA200 (bearish bias on 1d)
        bearish_setup = (rsi_14[i] > 70) and (close[i] < ema_1d_200_aligned[i])
        
        # === Exit Conditions (mean reversion complete) ===
        # Exit long when RSI returns to neutral (>50)
        # Exit short when RSI returns to neutral (<50)
        
        if bullish_setup:
            signals[i] = SIZE
        elif bearish_setup:
            signals[i] = -SIZE
        else:
            # Check for exit conditions on existing positions
            # We use a simple approach: if we have a position and RSI crosses 50, exit
            # Since we don't track position state explicitly, we rely on signal changes
            # A signal change from non-zero to zero (or vice versa) will trigger a trade
            # For mean reversion, we exit when RSI returns to 50
            # But to avoid whipsaw, we only exit if we were previously in a signal
            # However, to keep it simple and avoid look-ahead, we use:
            # If currently flat and RSI is near 50, stay flat
            # If we had a signal (non-zero) and RSI crosses 50, we'll naturally get signal=0 next bar
            # This is handled by the entry conditions failing
            signals[i] = 0.0
    
    return signals