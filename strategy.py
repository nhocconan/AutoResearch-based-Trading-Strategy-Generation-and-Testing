#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h and 1d Camarilla pivot levels with volume confirmation and ATR filter
# 4h/1d Camarilla levels (R3/S3, R4/S4) act as major support/resistance that work in both bull and bear markets
# Fade at R3/S3 (mean reversion), breakout continuation at R4/S4 (trend following)
# Volume confirmation (current 1h volume > 1.3x 20-period average) filters false signals
# ATR filter ensures sufficient volatility (avoid choppy low-vol periods)
# Position size fixed at 0.20 to minimize fee churn
# Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)

name = "1h_4h_1d_camarilla_atr_volume_v1"
timeframe = "1h"
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
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 10 or len(df_1d) < 10:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h Camarilla pivot levels
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    r4_4h = close_4h + range_4h * 1.1 / 2.0
    r3_4h = close_4h + range_4h * 1.1 / 4.0
    s3_4h = close_4h - range_4h * 1.1 / 4.0
    s4_4h = close_4h - range_4h * 1.1 / 2.0
    
    # Calculate 1d Camarilla pivot levels
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r4_1d = close_1d + range_1d * 1.1 / 2.0
    r3_1d = close_1d + range_1d * 1.1 / 4.0
    s3_1d = close_1d - range_1d * 1.1 / 4.0
    s4_1d = close_1d - range_1d * 1.1 / 2.0
    
    # Calculate 4h ATR (14-period) for volatility filtering
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr_14_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR (14-period) for volatility filtering
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]  # First period has no previous close
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF data to 1h timeframe
    r4_4h_aligned = align_htf_to_ltf(prices, df_4h, r4_4h)
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    s4_4h_aligned = align_htf_to_ltf(prices, df_4h, s4_4h)
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Pre-compute volume confirmation (20-period average for 1h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(r4_4h_aligned[i]) or np.isnan(r3_4h_aligned[i]) or
            np.isnan(s3_4h_aligned[i]) or np.isnan(s4_4h_aligned[i]) or
            np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(atr_4h_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(vol_ma_20[i]) or not in_session[i] or
            atr_4h_aligned[i] <= 0 or atr_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.3x average 1h volume
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        # Volatility filter: only trade when both 4h and 1d ATR are above their 50-period averages
        atr_ma_50_4h = pd.Series(atr_4h_aligned).rolling(window=50, min_periods=50).mean()
        atr_ma_50_1d = pd.Series(atr_1d_aligned).rolling(window=50, min_periods=50).mean()
        if len(atr_ma_50_4h) > i and len(atr_ma_50_1d) > i:
            vol_filter = (atr_4h_aligned[i] > atr_ma_50_4h.iloc[i]) and (atr_1d_aligned[i] > atr_ma_50_1d.iloc[i])
        else:
            vol_filter = True  # Not enough data for MA, allow trading
            
        if not vol_filter:
            signals[i] = 0.0
            continue
        
        # Fixed position size to minimize fee churn
        position_size = 0.20
        
        if position == 1:  # Long position
            # Exit on retracement to S3 (4h or 1d) or stop at S4 breakdown (4h or 1d)
            if close[i] < s3_4h_aligned[i] or close[i] < s3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] < s4_4h_aligned[i] or close[i] < s4_1d_aligned[i]:  # Stop loss at S4 breakdown
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on retracement to R3 (4h or 1d) or stop at R4 breakout (4h or 1d)
            if close[i] > r3_4h_aligned[i] or close[i] > r3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] > r4_4h_aligned[i] or close[i] > r4_1d_aligned[i]:  # Stop loss at R4 breakout
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Camarilla pivot trading with volume and volatility confirmation
            # Fade at R3/S3 (mean reversion), breakout at R4/S4 (trend following)
            if volume_confirmed:
                # Fade at R3 (sell at resistance, expect reversion to pivot)
                if (close[i] > r3_4h_aligned[i] and close[i] < r4_4h_aligned[i]) or \
                   (close[i] > r3_1d_aligned[i] and close[i] < r4_1d_aligned[i]):
                    position = -1
                    signals[i] = -position_size
                # Fade at S3 (buy at support, expect reversion to pivot)
                elif (close[i] < s3_4h_aligned[i] and close[i] > s4_4h_aligned[i]) or \
                     (close[i] < s3_1d_aligned[i] and close[i] > s4_1d_aligned[i]):
                    position = 1
                    signals[i] = position_size
                # Breakout continuation at R4 (buy break above resistance)
                elif close[i] > r4_4h_aligned[i] or close[i] > r4_1d_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Breakout continuation at S4 (sell break below support)
                elif close[i] < s4_4h_aligned[i] or close[i] < s4_1d_aligned[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals