#!/usr/bin/env python3
# Hypothesis: 1h RSI mean reversion with 4h trend filter and session filter (08-20 UTC).
# Long when RSI(14) < 30, price > 4h EMA50 (uptrend), and volume > 1.3x 20-period average, during 08-20 UTC.
# Short when RSI(14) > 70, price < 4h EMA50 (downtrend), and volume > 1.3x 20-period average, during 08-20 UTC.
# Uses ATR(14) trailing stop (1.5x) for risk control.
# RSI mean reversion works in ranging markets; 4h EMA50 filter ensures we trade with intermediate-term trend.
# Session filter reduces noise from low-volume off-hours. Target: 60-150 total trades over 4 years (15-37/year) on 1h.

name = "1h_RSI_MeanReversion_4hEMA50_Trend_Volume_Session_v1"
timeframe = "1h"
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA(50) on 4h data
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe (wait for 4h bar to close)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate volume confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma_20)
    
    # Session filter: 08-20 UTC (already datetime64[ms] in open_time column)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(atr[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: RSI < 30 (oversold) AND price > 4h EMA50 (uptrend) AND volume confirmation
            if rsi[i] < 30 and close[i] > ema_50_4h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: RSI > 70 (overbought) AND price < 4h EMA50 (downtrend) AND volume confirmation
            elif rsi[i] > 70 and close[i] < ema_50_4h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
                lowest_since_entry[i] = low[i]  # Initialize tracking
            else:
                signals[i] = 0.0
                # Carry forward tracking values when flat
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            # EXIT LONG: trailing stop hit (1.5x ATR)
            trailing_stop = close[i] < (highest_since_entry[i] - 1.5 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                highest_since_entry[i] = np.nan
            else:
                signals[i] = 0.20
                # Carry forward tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: trailing stop hit (1.5x ATR)
            trailing_stop = close[i] > (lowest_since_entry[i] + 1.5 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.20
                # Carry forward tracking
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals