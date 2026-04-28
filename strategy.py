#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 Breakout + 12h EMA50 Trend + Volume Spike
# Camarilla pivot levels provide precise intraday support/resistance.
# Breakout above R1 (resistance 1) with 12h EMA50 uptrend and volume spike = long.
# Breakdown below S1 (support 1) with 12h EMA50 downtrend and volume spike = short.
# Exit on retracement to pivot point (PP) or opposite Camarilla level.
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 75-200 total trades over 4 years (19-50/year).
# Camarilla levels are calculated from prior day's OHLC, making them inherently HTF.
# Works in both bull/bear markets by requiring alignment with 12h trend.
# Volume confirmation filters weak breakouts.

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter and Camarilla calculation (requires daily OHLC)
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from prior 12h bar (which represents prior day for 4h chart)
    # Camarilla uses prior period's OHLC: PP = (H+L+C)/3
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We shift by 1 to use prior completed 12h bar's OHLC
    prior_high = df_12h['high'].shift(1).values
    prior_low = df_12h['low'].shift(1).values
    prior_close = df_12h['close'].shift(1).values
    
    # Calculate Camarilla levels
    pp = (prior_high + prior_low + prior_close) / 3.0
    r1 = prior_close + (prior_high - prior_low) * 1.1 / 12.0
    s1 = prior_close - (prior_high - prior_low) * 1.1 / 12.0
    
    # Align Camarilla levels to 4h (they change only when 12h bar closes)
    pp_aligned = align_htf_to_ltf(prices, df_12h, pp)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure sufficient history for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 12h EMA trend filter
        ema_trend_up = close[i] > ema_50_12h_aligned[i]
        ema_trend_down = close[i] < ema_50_12h_aligned[i]
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Price > R1, 12h EMA50 uptrend, volume confirm
            if price > r1_aligned[i] and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: Price < S1, 12h EMA50 downtrend, volume confirm
            elif price < s1_aligned[i] and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on retracement to PP or below S1
            if price < pp_aligned[i] or price < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on retracement to PP or above R1
            if price > pp_aligned[i] or price > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals