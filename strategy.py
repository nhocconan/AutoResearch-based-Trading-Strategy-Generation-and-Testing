#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with volume confirmation and ATR volatility filter
# Uses 1d Camarilla levels (H3/L3) as entry triggers with volume > 1.5x 20-period average
# ATR(14) filter ensures trades only during sufficient volatility (> 30-period ATR mean)
# Fixed position size 0.25 to balance return and drawdown
# Target: 20-50 trades/year (80-200 total over 4 years) on 4h timeframe
# Works in bull via breakout continuation, in bear via mean reversion at extreme levels

name = "4h_1d_camarilla_breakout_volume_atr_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR (14-period) for volatility filtering
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d Camarilla levels (based on previous day)
    # Camarilla: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    # Using previous day's OHLC to avoid look-ahead
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = close_1d[0]  # First period uses current close
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    camarilla_width = 1.1 * (prev_high_1d - prev_low_1d) / 2.0
    h3_level = prev_close_1d + camarilla_width  # H3 resistance
    l3_level = prev_close_1d - camarilla_width  # L3 support
    
    # Align all HTF data to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_level)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_level)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Pre-compute volume confirmation (20-period average for 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or
            atr_14_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 4h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Volatility filter: only trade when 1d ATR is above its 30-period average
        atr_ma_30_1d = pd.Series(atr_14_1d_aligned).rolling(window=30, min_periods=30).mean()
        if len(atr_ma_30_1d) > i:
            vol_filter = atr_14_1d_aligned[i] > atr_ma_30_1d.iloc[i]
        else:
            vol_filter = True  # Not enough data for MA, allow trading
            
        if not vol_filter:
            signals[i] = 0.0
            continue
        
        # Fixed position size to minimize fee churn
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit on retracement to previous day's close (mean reversion)
            if close[i] <= prev_close_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on retracement to previous day's close (mean reversion)
            if close[i] >= prev_close_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Camarilla breakout with volume and volatility confirmation
            if volume_confirmed:
                # Breakout above H3 resistance (buy)
                if close[i] > h3_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Breakout below L3 support (sell)
                elif close[i] < l3_aligned[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals