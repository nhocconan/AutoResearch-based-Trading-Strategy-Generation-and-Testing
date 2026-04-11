#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_rsi_vwap_reversion"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h VWAP calculation
    vwap_num = (high + low + close) * volume
    vwap_den = volume
    vwap_cumsum_num = np.cumsum(vwap_num)
    vwap_cumsum_den = np.cumsum(vwap_den)
    vwap = vwap_cumsum_num / vwap_cumsum_den
    
    # 4h RSI (14 period)
    delta = np.diff(close)
    delta = np.concatenate([[0], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 4h ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily VWAP from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    vwap_num_1d = (high_1d + low_1d + close_1d) * volume_1d
    vwap_den_1d = volume_1d
    vwap_cumsum_num_1d = np.cumsum(vwap_num_1d)
    vwap_cumsum_den_1d = np.cumsum(vwap_den_1d)
    vwap_1d = vwap_cumsum_num_1d / vwap_cumsum_den_1d
    
    # Shift by 1 to use only completed daily bars
    vwap_1d = np.roll(vwap_1d, 1)
    vwap_1d[0] = np.nan
    
    # Align daily VWAP to 4h timeframe
    vwap_1d_4h = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Session filter: 0-23 UTC (4h bars cover full day)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 0) & (hours <= 23)
    
    # Minimum holding period: 4 bars (16 hours) to reduce churn
    hold_count = np.zeros(n, dtype=int)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(80, n):
        # Decrease hold counter
        if hold_count[i] > 0:
            hold_count[i] -= 1
        
        # Skip if any required data is invalid or outside session or holding
        if (np.isnan(vwap[i]) or np.isnan(rsi[i]) or
            np.isnan(atr[i]) or np.isnan(vwap_1d_4h[i]) or
            not in_session[i] or hold_count[i] > 0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        rsi_val = rsi[i]
        atr_val = atr[i]
        
        # Distance from VWAP in ATR units
        vwap_dist = (price_close - vwap[i]) / atr_val if atr_val > 0 else 0
        
        # Mean reversion conditions
        long_signal = (vwap_dist < -1.5) and (rsi_val < 30)
        short_signal = (vwap_dist > 1.5) and (rsi_val > 70)
        
        # Exit when price returns to VWAP or RSI neutralizes
        exit_long = position == 1 and ((vwap_dist > -0.5) or (rsi_val > 50))
        exit_short = position == -1 and ((vwap_dist < 0.5) or (rsi_val < 50))
        
        # Trading logic with minimum holding period
        if long_signal and position != 1:
            position = 1
            hold_count[i] = 4  # Hold for 4 bars minimum
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            hold_count[i] = 4  # Hold for 4 bars minimum
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 4h RSI-VWAP mean reversion strategy for BTC/ETH.
# Combines VWAP deviation with RSI extremes to identify mean reversion opportunities.
# Enters long when price is >1.5 ATR below VWAP and RSI < 30 (oversold).
# Enters short when price is >1.5 ATR above VWAP and RSI > 70 (overbought).
# Exits when price returns within 0.5 ATR of VWAP or RSI crosses 50.
# Uses daily VWAP as longer-term reference to align with institutional levels.
# Minimum holding period of 4 bars reduces churn and fee drag.
# Works in both bull and bear markets by capturing mean reversion moves.
# Target: 40-80 total trades over 4 years (10-20/year) to minimize fee drag.