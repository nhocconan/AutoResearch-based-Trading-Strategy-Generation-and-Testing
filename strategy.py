#!/usr/bin/env python3
# 6h_weekly_pivot_breakout_1d_atr_v1
# Hypothesis: Weekly pivot levels act as strong support/resistance on 6h timeframe.
# Breakouts above weekly R1 or below weekly S1 with volume confirmation and ATR filter
# capture institutional flow. 1d EMA50 filters trend direction to avoid counter-trend trades.
# Works in bull/bear: pivot levels adapt to volatility, EMA filter ensures trend alignment.
# Target: 15-25 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_breakout_1d_atr_v1"
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
    
    # 1d HTF data for EMA50 and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d ATR(14) for volatility filter
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Weekly pivot points from prior week (using 1d data)
    # Calculate weekly high/low/close from 1d data (resample to weekly)
    # We'll compute weekly pivot using the last completed week's OHLC
    # To avoid look-ahead, we use the weekly data from the prior week
    # We'll resample 1d data to weekly using pandas (but only once before loop)
    df_1d_indexed = pd.DataFrame({
        'open': df_1d['open'].values,
        'high': df_1d['high'].values,
        'low': df_1d['low'].values,
        'close': df_1d['close'].values
    }, index=pd.to_datetime(df_1d.index))
    
    # Resample to weekly (Friday close)
    df_weekly = df_1d_indexed.resample('W-FRI').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Get prior week's OHLC (shift by 1 to avoid look-ahead)
    weekly_open = df_weekly['open'].shift(1).values
    weekly_high = df_weekly['high'].shift(1).values
    weekly_low = df_weekly['low'].shift(1).values
    weekly_close = df_weekly['close'].shift(1).values
    
    # Calculate weekly pivot points
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    r2 = pp + (weekly_high - weekly_low)
    s2 = pp - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pp - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pp)
    
    # Align weekly pivot levels to 6h timeframe
    # We need to align the weekly values to the 6h index
    # Since weekly data is lower frequency, we forward fill the prior week's levels
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp, additional_delay_bars=0)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1, additional_delay_bars=0)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1, additional_delay_bars=0)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2, additional_delay_bars=0)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2, additional_delay_bars=0)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3, additional_delay_bars=0)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3, additional_delay_bars=0)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: close below weekly PP OR ATR expansion (stop loss)
            if close[i] < pp_aligned[i] or close[i] < low[i-1] - 1.5 * atr_14_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: close above weekly PP OR ATR expansion (stop loss)
            if close[i] > pp_aligned[i] or close[i] > high[i-1] + 1.5 * atr_14_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Long breakout: close above R1 with price above 1d EMA50
                if close[i] > r1_aligned[i] and close[i] > ema_50_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: close below S1 with price below 1d EMA50
                elif close[i] < s1_aligned[i] and close[i] < ema_50_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals