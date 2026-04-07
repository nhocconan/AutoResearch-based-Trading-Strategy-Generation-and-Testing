#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal with 12h trend filter and volume confirmation
# Long when price breaks below S3 with 12h EMA(50) bullish and volume > 1.5x 20-period average
# Short when price breaks above R3 with 12h EMA(50) bearish and volume > 1.5x 20-period average
# Exit when price crosses opposite Camarilla level (S4 for long, R4 for short)
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25
# Uses Camarilla levels from daily data for institutional reversal points

name = "6h_camarilla_reversal_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Using standard Camarilla formula: 
    # R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # S4 = C - ((H-L) * 1.1/2), S3 = C - ((H-L) * 1.1/4)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid look-ahead: use previous day's data
    camarilla_c = prev_close
    camarilla_range = prev_high - prev_low
    camarilla_r4 = camarilla_c + (camarilla_range * 1.1 / 2)
    camarilla_r3 = camarilla_c + (camarilla_range * 1.1 / 4)
    camarilla_s4 = camarilla_c - (camarilla_range * 1.1 / 2)
    camarilla_s3 = camarilla_c - (camarilla_range * 1.1 / 4)
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: 20-period average
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(ema_12h_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above S4 (stop and reverse condition)
            elif close[i] > camarilla_s4_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below R4 (stop and reverse condition)
            elif close[i] < camarilla_r4_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Camarilla S3/R3 break with 12h trend filter and volume confirmation
            # Volume filter: volume > 1.5x 20-period average
            volume_filter = volume[i] > 1.5 * volume_ma[i]
            # Trend filter: 12h EMA(50) - bullish if price > EMA, bearish if price < EMA
            trend_filter_long = close[i] > ema_12h_aligned[i]
            trend_filter_short = close[i] < ema_12h_aligned[i]
            
            # Long: price breaks below S3 (mean reversion) with bullish 12h trend and volume
            if close[i] < camarilla_s3_aligned[i] and trend_filter_long and volume_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks above R3 (mean reversion) with bearish 12h trend and volume
            elif close[i] > camarilla_r3_aligned[i] and trend_filter_short and volume_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals