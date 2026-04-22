#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4-hour price reversion at daily VWAP with 12-hour EMA50 trend filter and volume spike
    # Works in bull/bear via trend filter: only take long in uptrend, short in downtrend.
    # VWAP acts as dynamic support/resistance; trend filter ensures direction alignment; volume confirms.
    # Targets ~25 trades/year to minimize fee drag.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate typical price and cumulative VWAP components
    typical_price = (high + low + close) / 3.0
    tpv = typical_price * volume
    
    # Load 1d data for VWAP calculation (use previous day's data to avoid look-ahead)
    df_1d = get_htf_data(prices, '1d')
    tpv_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0 * df_1d['volume']
    vol_1d = df_1d['volume']
    
    # Calculate cumulative sums for VWAP using previous day's data
    cum_tpv = np.nancumsum(tpv_1d.values)
    cum_vol = np.nancumsum(vol_1d.values)
    
    # Calculate VWAP for each day (end of day value)
    vwap_1d = np.where(cum_vol != 0, cum_tpv / cum_vol, 0.0)
    # Use previous day's VWAP to avoid look-ahead
    vwap_1d_prev = np.roll(vwap_1d, 1)
    vwap_1d_prev[0] = np.nan  # First day has no previous
    
    # Align previous day's VWAP to 4h timeframe
    vwap_1d_prev_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d_prev)
    
    # Load 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume spike filter (20-period on 4h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(vwap_1d_prev_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or
            np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price crosses above VWAP (support) with volume + price above 12h EMA50 (uptrend)
            if close[i] > vwap_1d_prev_aligned[i] and vol_spike[i] and close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price crosses below VWAP (resistance) with volume + price below 12h EMA50 (downtrend)
            elif close[i] < vwap_1d_prev_aligned[i] and vol_spike[i] and close[i] < ema50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite side of VWAP or trend reversal vs 12h EMA50
            if position == 1:
                if close[i] < vwap_1d_prev_aligned[i] or close[i] < ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > vwap_1d_prev_aligned[i] or close[i] > ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_VWAP_Reversion_12hEMA50_Volume_Session_v1"
timeframe = "4h"
leverage = 1.0