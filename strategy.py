#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h trend following with 4h/1d confluence and session filter
# Long when 4h EMA21 rising + price above 4h EMA21 + 1d EMA50 up + 1h close > 1h VWAP + session (08-20 UTC)
# Short when 4h EMA21 falling + price below 4h EMA21 + 1d EMA50 down + 1h close < 1h VWAP + session (08-20 UTC)
# Uses multi-timeframe trend alignment to reduce whipsaw, VWAP for intraday timing, session filter to avoid low-volatility periods
# Targets 15-37 trades per year (60-150 over 4 years) for low fee drag
# Works in both bull and bear markets due to multi-timeframe trend filter

name = "1h_EMA21_VWAP_Session_Trend"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA21 for trend filter
    close_4h = df_4h['close'].values
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1h VWAP (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.nancumsum(pv)
    cum_vol = np.nancumsum(volume)
    vwap = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    
    # Precompute session hours (08-20 UTC) - 08:00 to 20:00 inclusive
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need sufficient data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema21_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vwap[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        ema21_4h_val = ema21_4h_aligned[i]
        ema50_1d_val = ema50_1d_aligned[i]
        vwap_val = vwap[i]
        in_session_val = in_session[i]
        
        if position == 0:
            # Enter long: 4h uptrend + price above 4h EMA21 + 1d uptrend + price above VWAP + in session
            if (close_val > ema21_4h_val and 
                ema21_4h_val > ema21_4h_aligned[i-1] and  # 4h EMA21 rising
                close_val > ema50_1d_val and
                ema50_1d_val > ema50_1d_aligned[i-1] and  # 1d EMA50 rising
                close_val > vwap_val and
                in_session_val):
                signals[i] = 0.20
                position = 1
            # Enter short: 4h downtrend + price below 4h EMA21 + 1d downtrend + price below VWAP + in session
            elif (close_val < ema21_4h_val and
                  ema21_4h_val < ema21_4h_aligned[i-1] and  # 4h EMA21 falling
                  close_val < ema50_1d_val and
                  ema50_1d_val < ema50_1d_aligned[i-1] and  # 1d EMA50 falling
                  close_val < vwap_val and
                  in_session_val):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: 4h trend down OR price below 4h EMA21
            if (close_val < ema21_4h_val or 
                ema21_4h_val < ema21_4h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: 4h trend up OR price above 4h EMA21
            if (close_val > ema21_4h_val or 
                ema21_4h_val > ema21_4h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals