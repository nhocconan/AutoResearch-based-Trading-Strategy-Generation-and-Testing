#!/usr/bin/env python3
# 1d_Keltner_Breakout_1wTrend_Volume
# Hypothesis: On 1d timeframe, use Keltner Channel breakouts with 1-week EMA trend filter and volume confirmation.
# Enter long when price closes above upper Keltner band (ATR-based) with volume > 1.8x 20-day average and 1w EMA uptrend.
# Enter short when price closes below lower Keltner band with volume > 1.8x and 1w EMA downtrend.
# Exit when price crosses the 1-week EMA (trend reversal).
# Targets 15-25 trades/year to minimize fee drift while capturing multi-week trends in both bull and bear markets.
# Position sizing: 0.25 for standard, 0.35 when volume > 2.5x average.

name = "1d_Keltner_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    
    # 1-week EMA21 for trend filter
    ema21_1w = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # ATR for Keltner Channel (20-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel: EMA20 ± 2 * ATR
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema20 + 2 * atr
    lower_keltner = ema20 - 2 * atr
    
    # Volume confirmation: 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema21_1w_aligned[i]) or np.isnan(upper_keltner[i]) or 
            np.isnan(lower_keltner[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        ema1w_trend = ema21_1w_aligned[i]
        upper_kc = upper_keltner[i]
        lower_kc = lower_keltner[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # LONG: Price closes above upper Keltner with volume and 1w uptrend
            if close[i] > upper_kc and close[i] > ema1w_trend and vol_ratio_val > 1.8:
                # Dynamic sizing: increase on extreme volume
                if vol_ratio_val > 2.5:
                    signals[i] = 0.35
                else:
                    signals[i] = 0.25
                position = 1
            # SHORT: Price closes below lower Keltner with volume and 1w downtrend
            elif close[i] < lower_kc and close[i] < ema1w_trend and vol_ratio_val > 1.8:
                if vol_ratio_val > 2.5:
                    signals[i] = -0.35
                else:
                    signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 1-week EMA (trend reversal)
            if close[i] < ema1w_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if vol_ratio_val <= 2.5 else 0.35
        elif position == -1:
            # EXIT SHORT: Price crosses above 1-week EMA (trend reversal)
            if close[i] > ema1w_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25 if vol_ratio_val <= 2.5 else -0.35
    
    return signals