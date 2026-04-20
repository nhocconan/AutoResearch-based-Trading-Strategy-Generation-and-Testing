# 12h_1d_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1
# Hypothesis: Price tends to revert to the mean from Camarilla pivot levels (R1, S1) with volume confirmation.
# In both bull and bear markets, price often retraces to these levels before continuing the trend.
# Using 1d Camarilla levels for context, with 12h entries, reduces whipsaw vs lower timeframes.
# Volume filter ensures institutional participation, ATR filter avoids extreme volatility.
# Target: 20-50 trades per year per symbol, low frequency to minimize fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivot levels, trend, volatility, volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d Camarilla pivot levels (based on previous day)
    # R4 = close + (high - low) * 1.500
    # R3 = close + (high - low) * 1.250
    # R2 = close + (high - low) * 1.166
    # R1 = close + (high - low) * 1.083
    # PP = (high + low + close) / 3
    # S1 = close - (high - low) * 1.083
    # S2 = close - (high - low) * 1.166
    # S3 = close - (high - low) * 1.250
    # S4 = close - (high - low) * 1.500
    # We use R1 and S1 as key reversal levels
    
    # Calculate using shifted previous day's data to avoid look-ahead
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # first bar uses current
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    camarilla_width = prev_high - prev_low
    r1 = prev_close + camarilla_width * 1.083
    s1 = prev_close - camarilla_width * 1.083
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d EMA(34) for trend filter (as seen in top performers)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d ATR(14) for volatility filtering and stop
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 1d Volume ratio (current / 20-period average) for confirmation
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / np.where(vol_ma_20_1d == 0, 1, vol_ma_20_1d)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ratio_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_trend = ema_34_1d_aligned[i]
        atr = atr_14_1d_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        vol_ratio = vol_ratio_1d_aligned[i]
        
        # Trend filter: price relative to daily EMA
        trend_up = price > ema_trend
        trend_down = price < ema_trend
        
        # Volatility filter: avoid extreme volatility spikes
        atr_ma_20 = pd.Series(atr_14_1d_aligned).rolling(window=20, min_periods=20).mean().values[i]
        vol_filter = (atr > 0.3 * atr_ma_20) and (atr < 3.0 * atr_ma_20)
        
        # Volume filter: require above-average volume
        vol_filter = vol_filter and (vol_ratio > 1.3)
        
        if position == 0:
            # Enter long near S1 in uptrend with volume confirmation
            if trend_up and vol_filter and price <= s1_level * 1.002:  # within 0.2% of S1
                signals[i] = 0.25
                position = 1
            # Enter short near R1 in downtrend with volume confirmation
            elif trend_down and vol_filter and price >= r1_level * 0.998:  # within 0.2% of R1
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend reversal, or price reaches R1 (take profit)
            if not trend_up or price >= r1_level * 0.998:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend reversal, or price reaches S1 (take profit)
            if not trend_down or price <= s1_level * 1.002:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1"
timeframe = "12h"
leverage = 1.0