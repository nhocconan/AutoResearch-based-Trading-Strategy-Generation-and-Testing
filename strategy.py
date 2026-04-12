#!/usr/bin/env python3
"""
1d_1w_Weekly_Pullback_to_Trend_v1
Hypothesis: Price pulls back to weekly trend (EMA21) during ranging markets (low volatility) and resumes trend. Uses daily chart for entry timing.
Long when price touches weekly EMA21 during low volatility (weekly ATR < 50th percentile) and daily RSI < 40.
Short when price touches weekly EMA21 during low volatility and daily RSI > 60.
Uses volatility regime to avoid whipsaws in high volatility markets.
Targets 10-25 trades/year to minimize fee drag.
Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Weekly_Pullback_to_Trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend and volatility regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # === WEEKLY EMA21 (trend) ===
    ema21 = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_daily = align_htf_to_ltf(prices, df_1w, ema21)
    
    # === WEEKLY ATR(14) for volatility regime ===
    tr1 = weekly_high - weekly_low
    tr2 = np.abs(weekly_high[1:] - weekly_close[:-1])
    tr3 = np.abs(weekly_low[1:] - weekly_close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr14_daily = align_htf_to_ltf(prices, df_1w, atr14)
    
    # === WEEKLY ATR PERCENTILE (50-day lookback) ===
    atr_percentile = np.full(n, np.nan)
    if n >= 50:
        for i in range(50, n):
            window = atr14_daily[i-50:i]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                percentile = (np.sum(valid <= atr14_daily[i]) / len(valid)) * 100
                atr_percentile[i] = percentile
    
    # === DAILY RSI(14) for entry timing ===
    delta = pd.Series(close).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(ema21_daily[i]) or np.isnan(atr14_daily[i]) or 
            np.isnan(atr_percentile[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Low volatility regime: weekly ATR below 50th percentile
        low_vol = atr_percentile[i] < 50
        
        # Price near weekly EMA21 (within 0.5%)
        price_near_ema = np.abs(close[i] - ema21_daily[i]) / ema21_daily[i] < 0.005
        
        # Entry conditions
        long_entry = low_vol and price_near_ema and (rsi[i] < 40)
        short_entry = low_vol and price_near_ema and (rsi[i] > 60)
        
        # Exit: volatility increases or price moves away from EMA
        high_vol = atr_percentile[i] > 70
        price_far = np.abs(close[i] - ema21_daily[i]) / ema21_daily[i] > 0.02
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and (high_vol or price_far):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (high_vol or price_far):
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals