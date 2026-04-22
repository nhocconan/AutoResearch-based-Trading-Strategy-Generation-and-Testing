#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4-hour Bollinger Bands breakout with 1-day ATR filter and volume confirmation
    # Works in bull/bear via ATR-based volatility filter: only trade when volatility is elevated.
    # Bollinger breakouts capture momentum; ATR filter avoids low-volatility chop; volume confirms.
    # Targets ~20-30 trades/year to minimize fee drag.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands on 4h (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + bb_std * bb_std_dev
    lower_band = sma - bb_std * bb_std_dev
    
    # 1-day ATR for volatility filter (14-period)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # First element will be invalid due to roll, handled by min_periods
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR-based volatility filter: only trade when ATR > 20-period SMA of ATR
    atr_ma = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    vol_filter = atr_14 > atr_ma  # Trade only when volatility is above average
    
    # Align ATR filter to 4h timeframe
    vol_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_filter)
    
    # Volume confirmation (20-period on 4h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Require 1.5x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(vol_filter_aligned[i]) or np.isnan(vol_ma20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close crosses above upper Bollinger Band with volatility filter + volume spike
            if close[i] > upper_band[i] and vol_filter_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close crosses below lower Bollinger Band with volatility filter + volume spike
            elif close[i] < lower_band[i] and vol_filter_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Close returns to middle Bollinger Band (SMA) or volatility drops
            if position == 1:
                if close[i] < sma[i] or not vol_filter_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > sma[i] or not vol_filter_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Bollinger_Breakout_ATRVolFilter_Volume_Session_v1"
timeframe = "4h"
leverage = 1.0