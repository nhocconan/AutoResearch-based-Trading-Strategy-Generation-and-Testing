#!/usr/bin/env python3
# Hypothesis: 1h strategy using 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above 4h Camarilla R1 AND 1d EMA34 > prior 1d EMA34 (uptrend) AND volume > 1.5x 20-period average.
# Short when price breaks below 4h Camarilla S1 AND 1d EMA34 < prior 1d EMA34 (downtrend) AND volume > 1.5x 20-period average.
# Uses ATR(14) trailing stop (2.0x) for risk control.
# 4h Camarilla levels provide precise intraday support/resistance that work in ranging and trending markets.
# 1d EMA34 trend filter ensures we trade with the longer-term trend, reducing whipsaws in bear markets.
# Volume confirmation adds validity to breakouts. Session filter (08-20 UTC) reduces noise trades.
# Target: 60-150 total trades over 4 years (15-37/year) on 1h.
# Signal size: 0.20 (discrete level to minimize fee churn).

name = "1h_Camarilla_R1S1_Breakout_1dEMA34_Trend_Volume_Session_v1"
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
    
    # Get 4h data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    open_4h = df_4h['open'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels: R1, S1 (based on previous 4h bar's OHLC)
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_r1 = close_4h + 1.1 * (high_4h - low_4h) / 12
    camarilla_s1 = close_4h - 1.1 * (high_4h - low_4h) / 12
    
    # Align Camarilla levels to 1h timeframe (wait for 4h bar to close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d data
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 1h timeframe (wait for 1d bar to close)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # Session filter: 08-20 UTC (inclusive)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]) or not session_filter[i]):
            signals[i] = 0.0
            # Carry forward tracking values when flat or skipped
            if i > 0 and position == 0:
                highest_since_entry[i] = highest_since_entry[i-1]
                lowest_since_entry[i] = lowest_since_entry[i-1]
            elif i > 0 and position == 1:
                highest_since_entry[i] = highest_since_entry[i-1]
            elif i > 0 and position == -1:
                lowest_since_entry[i] = lowest_since_entry[i-1]
            continue
        
        if position == 0:
            # LONG: Price > Camarilla R1 AND 1d EMA34 rising (trending up) AND volume confirmation
            if close[i] > camarilla_r1_aligned[i] and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Price < Camarilla S1 AND 1d EMA34 falling (trending down) AND volume confirmation
            elif close[i] < camarilla_s1_aligned[i] and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and volume_confirm[i]:
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
            # EXIT LONG: trailing stop hit (2.0x ATR)
            trailing_stop = close[i] < (highest_since_entry[i] - 2.0 * atr[i])
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
            # EXIT SHORT: trailing stop hit (2.0x ATR)
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.0 * atr[i])
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