#!/usr/bin/env python3
"""
1h_Camarilla_Pivot_Support_Resistance_Momentum
Hypothesis: Camarilla pivot levels (S3, S4, R3, R4) act as strong support/resistance in ranging markets, with momentum confirmation (RSI) and trend filter (4h EMA50) to avoid whipsaws. Enter long at S3/S4 with bullish momentum and 4h uptrend; enter short at R3/R4 with bearish momentum and 4h downtrend. Exit on opposite S1/R1 touch or momentum reversal. Uses 1d trend filter for higher timeframe bias. Session filter (08-20 UTC) reduces noise. Target: 20-40 trades/year per symbol.
"""

name = "1h_Camarilla_Pivot_Support_Resistance_Momentum"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels from previous day's range
    # Requires daily OHLC; we'll approximate using rolling window
    # For simplicity, we use previous 24h high/low (since 1h timeframe)
    prev_high = np.roll(high, 24)  # Previous day's high (24 periods back)
    prev_low = np.roll(low, 24)    # Previous day's low
    prev_close = np.roll(close, 24) # Previous day's close
    
    # Handle first 24 bars
    prev_high[:24] = high[:24]
    prev_low[:24] = low[:24]
    prev_close[:24] = close[:24]
    
    range_hl = prev_high - prev_low
    camarilla_mult = 1.1 / 12  # Camarilla multiplier
    
    # Calculate S1, S2, S3, S4 and R1, R2, R3, R4
    S4 = close - (range_hl * camarilla_mult * 6)
    S3 = close - (range_hl * camarilla_mult * 4)
    S2 = close - (range_hl * camarilla_mult * 2)
    S1 = close - (range_hl * camarilla_mult * 1)
    R1 = close + (range_hl * camarilla_mult * 1)
    R2 = close + (range_hl * camarilla_mult * 2)
    R3 = close + (range_hl * camarilla_mult * 4)
    R4 = close + (range_hl * camarilla_mult * 6)
    
    # RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h trend: EMA50
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    uptrend_4h = close > ema_50_4h_aligned
    downtrend_4h = close < ema_50_4h_aligned
    
    # 1d trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    uptrend_1d = close > ema_50_1d_aligned
    downtrend_1d = close < ema_50_1d_aligned
    
    # Volume confirmation: volume > 1.5 * 24-period average
    vol_ma = np.zeros(n)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    volume_conf = volume > 1.5 * vol_ma
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if not session_filter[i]:
            signals[i] = 0.0
            continue
            
        # Get values
        rsi_val = rsi[i]
        vol_conf = volume_conf[i]
        uptrend_htf = uptrend_1d[i]
        downtrend_htf = downtrend_1d[i]
        uptrend_4h_val = uptrend_4h[i]
        downtrend_4h_val = downtrend_4h[i]
        
        if position == 0:
            # LONG: price at S3/S4, bullish momentum (RSI > 50), 4h uptrend, 1d uptrend filter, volume
            if ((close[i] <= S3[i] or close[i] <= S4[i]) and 
                rsi_val > 50 and 
                uptrend_4h_val and 
                uptrend_htf and 
                vol_conf):
                signals[i] = 0.20
                position = 1
            # SHORT: price at R3/R4, bearish momentum (RSI < 50), 4h downtrend, 1d downtrend filter, volume
            elif ((close[i] >= R3[i] or close[i] >= R4[i]) and 
                  rsi_val < 50 and 
                  downtrend_4h_val and 
                  downtrend_htf and 
                  vol_conf):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: touch S1 or RSI < 40 (momentum loss) or 4h trend turns down
            if (close[i] >= S1[i] or rsi_val < 40 or not uptrend_4h_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: touch R1 or RSI > 60 (momentum loss) or 4h trend turns up
            if (close[i] <= R1[i] or rsi_val > 60 or not downtrend_4h_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals