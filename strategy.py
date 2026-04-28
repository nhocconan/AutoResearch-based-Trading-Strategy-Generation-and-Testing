#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 Breakout + 4h EMA50 Trend + Volume Spike + Session Filter (08-20 UTC)
# Uses 4h EMA50 for trend filter, prior day's OHLC for Camarilla levels (inherently 1d HTF),
# volume confirmation to filter weak breakouts, and session filter to avoid low-liquidity hours.
# Discrete position sizing (0.20) to limit drawdown and reduce fee churn.
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
# Works in both bull/bear markets by requiring alignment with 4h trend.

name = "1h_Camarilla_R1_S1_Breakout_4hEMA50_Trend_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter and 1d data for Camarilla calculation (requires daily OHLC)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from prior 1d bar
    # Camarilla uses prior period's OHLC: PP = (H+L+C)/3
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We shift by 1 to use prior completed 1d bar's OHLC
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    prior_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    pp = (prior_high + prior_low + prior_close) / 3.0
    r1 = prior_close + (prior_high - prior_low) * 1.1 / 12.0
    s1 = prior_close - (prior_high - prior_low) * 1.1 / 12.0
    
    # Align Camarilla levels to 1h (they change only when 1d bar closes)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure sufficient history for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 4h EMA trend filter
        ema_trend_up = close[i] > ema_50_4h_aligned[i]
        ema_trend_down = close[i] < ema_50_4h_aligned[i]
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Price > R1, 4h EMA50 uptrend, volume confirm
            if price > r1_aligned[i] and ema_trend_up and vol_confirm:
                signals[i] = 0.20
                position = 1
            # Short entry: Price < S1, 4h EMA50 downtrend, volume confirm
            elif price < s1_aligned[i] and ema_trend_down and vol_confirm:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on retracement to PP or below S1
            if price < pp_aligned[i] or price < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short - exit on retracement to PP or above R1
            if price > pp_aligned[i] or price > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals