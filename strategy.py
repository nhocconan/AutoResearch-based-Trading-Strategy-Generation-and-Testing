#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h trend filter and volume confirmation.
# Long when Bull Power > 0 (close > EMA13), Bear Power < 0 (low < EMA13), 12h EMA50 uptrend, and volume > 1.5x 20 EMA.
# Short when Bear Power < 0, Bull Power < 0, 12h EMA50 downtrend, and volume > 1.5x 20 EMA.
# Uses 13-period EMA for Elder Ray to capture short-term momentum, 12h EMA50 for trend filter, and volume spike for confirmation.
# Designed for moderate trade frequency (target: 20-40/year) with strong trend alignment to reduce whipsaws.
# Works in bull markets via long signals in uptrends and bear markets via short signals in downtrends.
name = "6h_ElderRay_12Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA13 for Elder Ray (13-period)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = Close - EMA13
    bull_power = close - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # Load 12h data for trend filter (EMA50) and volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 12h volume > 1.5x 20 EMA for confirmation
    vol_ema20_12h = pd.Series(df_12h['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio_12h = np.where(vol_ema20_12h > 0, df_12h['volume'].values / vol_ema20_12h, 1.0)
    vol_confirm_12h = vol_ratio_12h > 1.5
    vol_confirm_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_confirm_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA50
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_confirm_12h_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long condition: Bull Power > 0, Bear Power < 0, 12h uptrend, volume confirmation
            long_cond = (bull_power[i] > 0) and (bear_power[i] < 0) and (close[i] > ema50_12h_aligned[i]) and vol_confirm_12h_aligned[i]
            # Short condition: Bear Power < 0, Bull Power < 0, 12h downtrend, volume confirmation
            short_cond = (bear_power[i] < 0) and (bull_power[i] < 0) and (close[i] < ema50_12h_aligned[i]) and vol_confirm_12h_aligned[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: trend breakdown (close < EMA50) or loss of bull power (Bull Power <= 0)
            if (close[i] < ema50_12h_aligned[i]) or (bull_power[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: trend breakdown (close > EMA50) or loss of bear power (Bear Power >= 0)
            if (close[i] > ema50_12h_aligned[i]) or (bear_power[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals