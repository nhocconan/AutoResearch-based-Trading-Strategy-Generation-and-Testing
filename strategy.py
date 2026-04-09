#!/usr/bin/env python3
# 6h_12h_1d_camarilla_pullback_v1
# Hypothesis: 6h strategy using 12h Camarilla pivot levels with 1d trend filter and volume confirmation.
# Long: Price breaks above 12h Camarilla R3 level, pulls back to touch or cross above 6h EMA(20), with volume > 1.3x 20-period average and 1d close > 1d EMA(50).
# Short: Price breaks below 12h Camarilla S3 level, pulls back to touch or cross below 6h EMA(20), with volume > 1.3x 20-period average and 1d close < 1d EMA(50).
# Exit: Opposite Camarilla break (R4/S4) or ATR trailing stop (2.5x ATR from extreme).
# Uses 12h Camarilla for structure, 6h EMA for pullback entry, 1d EMA for trend filter, volume for confirmation, ATR for dynamic stops.
# Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_camarilla_pullback_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for volatility filter and trailing stop
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift()).abs()
    tr3 = (low_s - close_s.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Get 12h data for Camarilla pivot levels (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) == 0:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels
    high_12h = pd.Series(df_12h['high'].values)
    low_12h = pd.Series(df_12h['low'].values)
    close_12h = pd.Series(df_12h['close'].values)
    pivot = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    camarilla_r3 = pivot + range_12h * 1.1 / 4
    camarilla_s3 = pivot - range_12h * 1.1 / 4
    camarilla_r4 = pivot + range_12h * 1.1 / 2
    camarilla_s4 = pivot - range_12h * 1.1 / 2
    
    # Align HTF Camarilla levels to 6h timeframe (wait for completed 12h bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3.values)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4.values)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4.values)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 6h EMA(20) for pullback entry
    close_s = pd.Series(close)
    ema20_6h = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    long_high = 0.0   # highest high since long entry
    short_low = 0.0   # lowest low since short entry
    long_triggered = False  # flag to wait for pullback after breakout
    short_triggered = False  # flag to wait for pullback after breakout
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(atr[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(open_price[i]) or np.isnan(volume[i]) or np.isnan(ema20_6h[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            long_high = max(long_high, high[i])
            # ATR trailing stop: exit if price drops 2.5*ATR from high
            if long_high > 0 and close[i] < long_high - 2.5 * atr[i]:
                position = 0
                long_high = 0.0
                long_triggered = False
                signals[i] = 0.0
            # Exit: Price breaks above 12h Camarilla R4 (continuation break)
            elif close[i] > camarilla_r4_aligned[i]:
                position = 0
                long_high = 0.0
                long_triggered = False
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            short_low = min(short_low, low[i])
            # ATR trailing stop: exit if price rises 2.5*ATR from low
            if short_low > 0 and close[i] > short_low + 2.5 * atr[i]:
                position = 0
                short_low = 0.0
                short_triggered = False
                signals[i] = 0.0
            # Exit: Price breaks below 12h Camarilla S4 (continuation break)
            elif close[i] < camarilla_s4_aligned[i]:
                position = 0
                short_low = 0.0
                short_triggered = False
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for breakout and set trigger flags
            bullish_breakout = (close[i] > camarilla_r3_aligned[i]) and volume_confirmed and (close[i] > ema50_1d_aligned[i])
            bearish_breakout = (close[i] < camarilla_s3_aligned[i]) and volume_confirmed and (close[i] < ema50_1d_aligned[i])
            
            if bullish_breakout:
                long_triggered = True
                short_triggered = False
            elif bearish_breakout:
                short_triggered = True
                long_triggered = False
            
            # Long entry: after bullish breakout, price pulls back to EMA20 or above
            if long_triggered and close[i] >= ema20_6h[i]:
                position = 1
                long_high = high[i]
                long_triggered = False
                signals[i] = 0.25
            # Short entry: after bearish breakout, price pulls back to EMA20 or below
            elif short_triggered and close[i] <= ema20_6h[i]:
                position = -1
                short_low = low[i]
                short_triggered = False
                signals[i] = -0.25
    
    return signals