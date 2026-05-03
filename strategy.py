#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Camarilla pivot levels provide precise intraday support/resistance. Breakout above R3 or below S3
# with volume confirmation indicates strong momentum. 1d EMA34 filters for higher timeframe trend
# alignment to avoid counter-trend trades. This strategy aims for low-frequency, high-conviction
# trades suitable for 12h timeframe (target: 12-37 trades/year).

name = "12h_Camarilla_R3S3_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 1d Camarilla levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range for Camarilla calculation
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]  # First bar
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5*(high-low), R3 = close + 1.25*(high-low)
    # S3 = close - 1.25*(high-low), S4 = close - 1.5*(high-low)
    camarilla_range = high_1d - low_1d
    r3 = close_1d + 1.25 * camarilla_range
    s3 = close_1d - 1.25 * camarilla_range
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3, additional_delay_bars=1)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3, additional_delay_bars=1)
    
    # Calculate volume regime: current 12h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        close_val = close[i]
        vol_spike = volume_spike[i]
        ema_trend = ema_34_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        
        # Skip if any value is NaN
        if np.isnan(close_val) or np.isnan(vol_spike) or np.isnan(ema_trend) or \
           np.isnan(r3_val) or np.isnan(s3_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine trend regime: bull if close > 1d EMA34, bear if close < 1d EMA34
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Generate signals
        if position == 0:
            # Long entry: price breaks above R3 with volume spike in bull trend
            if is_bull_trend and close_val > r3_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 with volume spike in bear trend
            elif is_bear_trend and close_val < s3_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below 1d EMA34 or loses momentum (close < R3)
            if close_val < ema_trend or close_val < r3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above 1d EMA34 or loses momentum (close > S3)
            if close_val > ema_trend or close_val > s3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals