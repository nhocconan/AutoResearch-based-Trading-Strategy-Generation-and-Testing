#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h/1d Camarilla pivot levels with volume confirmation and ATR filter
# 12h/1d Camarilla levels (R3/S3, R4/S4) act as major support/resistance that work in both bull and bear markets
# Fade at R3/S3 (mean reversion), breakout continuation at R4/S4 (trend following)
# Volume confirmation (current 4h volume > 1.3x 20-period average) filters false signals
# ATR filter ensures sufficient volatility (avoid choppy low-vol periods)
# Position size fixed at 0.25 to balance risk and return
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)

name = "4h_12h_1d_camarilla_atr_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 10 or len(df_1d) < 10:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h Camarilla pivot levels
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    r4_12h = close_12h + range_12h * 1.1 / 2.0
    r3_12h = close_12h + range_12h * 1.1 / 4.0
    s3_12h = close_12h - range_12h * 1.1 / 4.0
    s4_12h = close_12h - range_12h * 1.1 / 2.0
    
    # Calculate 1d Camarilla pivot levels
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r4_1d = close_1d + range_1d * 1.1 / 2.0
    r3_1d = close_1d + range_1d * 1.1 / 4.0
    s3_1d = close_1d - range_1d * 1.1 / 4.0
    s4_1d = close_1d - range_1d * 1.1 / 2.0
    
    # Calculate 4h ATR (14-period) for volatility filtering
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr_14_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF data to 4h timeframe
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    atr_4h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_4h)  # Use 12h for alignment stability
    
    # Pre-compute volume confirmation (20-period average for 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(r4_12h_aligned[i]) or np.isnan(r3_12h_aligned[i]) or
            np.isnan(s3_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or
            np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(atr_4h_aligned[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i] or atr_4h_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.3x average 4h volume
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        # Volatility filter: only trade when ATR is above its 50-period average
        atr_ma_50 = pd.Series(atr_4h_aligned).rolling(window=50, min_periods=50).mean()
        if len(atr_ma_50) > i:
            vol_filter = atr_4h_aligned[i] > atr_ma_50.iloc[i]
        else:
            vol_filter = True  # Not enough data for MA, allow trading
            
        if not vol_filter:
            signals[i] = 0.0
            continue
        
        # Fixed position size to minimize fee churn
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit on retracement to S3 (12h or 1d) or stop at S4 breakdown (12h or 1d)
            if close[i] < s3_12h_aligned[i] or close[i] < s3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] < s4_12h_aligned[i] or close[i] < s4_1d_aligned[i]:  # Stop loss at S4 breakdown
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on retracement to R3 (12h or 1d) or stop at R4 breakout (12h or 1d)
            if close[i] > r3_12h_aligned[i] or close[i] > r3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] > r4_12h_aligned[i] or close[i] > r4_1d_aligned[i]:  # Stop loss at R4 breakout
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Camarilla pivot trading with volume and volatility confirmation
            # Fade at R3/S3 (mean reversion), breakout at R4/S4 (trend following)
            if volume_confirmed:
                # Fade at R3 (sell at resistance, expect reversion to pivot)
                if (close[i] > r3_12h_aligned[i] and close[i] < r4_12h_aligned[i]) or \
                   (close[i] > r3_1d_aligned[i] and close[i] < r4_1d_aligned[i]):
                    position = -1
                    signals[i] = -position_size
                # Fade at S3 (buy at support, expect reversion to pivot)
                elif (close[i] < s3_12h_aligned[i] and close[i] > s4_12h_aligned[i]) or \
                     (close[i] < s3_1d_aligned[i] and close[i] > s4_1d_aligned[i]):
                    position = 1
                    signals[i] = position_size
                # Breakout continuation at R4 (buy break above resistance)
                elif close[i] > r4_12h_aligned[i] or close[i] > r4_1d_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Breakout continuation at S4 (sell break below support)
                elif close[i] < s4_12h_aligned[i] or close[i] < s4_1d_aligned[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals