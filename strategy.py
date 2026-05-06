#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h session-based mean reversion with 4h/1d regime filter
# Uses 4h Supertrend for trend regime (bull/bear), 1d RSI for overextension filter
# Enters mean reversion on 1h during London/NY session (08-20 UTC) when price deviates from 4h VWAP
# Volume confirmation (>1.3x 20-bar average) reduces false signals
# Discrete sizing 0.20 to limit fee drag; target 80-150 total trades over 4 years (20-37.5/year)
# Works in bull markets (mean reversion in uptrend) and bear markets (mean reversion in downtrend)

name = "1h_SessionMeanReversion_4hSupertrend_1dRSI_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) - open_time is already datetime64[ms]
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h Supertrend (regime filter)
    hl2_4h = (df_4h['high'] + df_4h['low']) / 2
    atr_4h = pd.Series(
        np.maximum(
            np.maximum(df_4h['high'] - df_4h['low'],
                       np.abs(df_4h['high'] - df_4h['close'].shift(1))),
            np.abs(df_4h['low'] - df_4h['close'].shift(1))
        )
    ).rolling(window=10, min_periods=10).mean()
    
    upper_4h = hl2_4h + (3.0 * atr_4h)
    lower_4h = hl2_4h - (3.0 * atr_4h)
    
    supertrend_4h = np.zeros(len(df_4h))
    direction_4h = np.ones(len(df_4h))  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(df_4h)):
        if close_4h := df_4h['close'].iloc[i]:
            if supertrend_4h[i-1] == upper_4h[i-1]:
                if close_4h <= upper_4h[i]:
                    supertrend_4h[i] = upper_4h[i]
                else:
                    supertrend_4h[i] = lower_4h[i]
                    direction_4h[i] = -1
            else:
                if close_4h >= lower_4h[i]:
                    supertrend_4h[i] = lower_4h[i]
                else:
                    supertrend_4h[i] = upper_4h[i]
                    direction_4h[i] = 1
        else:
            supertrend_4h[i] = supertrend_4h[i-1]
            direction_4h[i] = direction_4h[i-1]
    
    # 1d RSI (overextension filter)
    close_1d = df_1d['close']
    delta = close_1d.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.fillna(50).values
    
    # 1h VWAP (mean reversion target)
    typical_price = (high + low + close) / 3
    vwap_num = (typical_price * volume).cumsum()
    vwap_den = volume.cumsum()
    vwap = vwap_num / vwap_den
    
    # Volume confirmation (>1.3x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_filter = volume > (1.3 * vol_ma_20)
    
    # Align HTF indicators to 1h timeframe
    direction_4h_aligned = align_htf_to_ltf(prices, df_4h, direction_4h.values)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap.values)  # 1d VWAP as reference
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(direction_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vwap_aligned[i]) or np.isnan(volume_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long mean reversion: price < VWAP AND 4h uptrend AND 1d not oversold AND volume spike
            if (close[i] < vwap_aligned[i] and 
                direction_4h_aligned[i] == 1 and 
                rsi_1d_aligned[i] > 30 and 
                volume_filter_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short mean reversion: price > VWAP AND 4h downtrend AND 1d not overbought AND volume spike
            elif (close[i] > vwap_aligned[i] and 
                  direction_4h_aligned[i] == -1 and 
                  rsi_1d_aligned[i] < 70 and 
                  volume_filter_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price >= VWAP (mean reversion complete) OR 4h trend changes
            if close[i] >= vwap_aligned[i] or direction_4h_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price <= VWAP (mean reversion complete) OR 4h trend changes
            if close[i] <= vwap_aligned[i] or direction_4h_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals