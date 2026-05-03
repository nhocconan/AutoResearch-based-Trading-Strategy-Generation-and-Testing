#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume spike confirmation.
# Camarilla pivots provide precise intraday support/resistance levels. Breakout above R3 or below S3
# indicates strong momentum. In bull regime (price > 1w EMA34), we go long on R3 breakout with volume spike.
# In bear regime (price < 1w EMA34), we go short on S3 breakdown with volume spike.
# This strategy adapts to both bull and bear markets by using the higher timeframe trend for regime filtering
# and Camarilla breakouts for precise entry timing. Target timeframe: 12h.

name = "12h_Camarilla_R3S3_1wTrend_VolumeSpike_Regime"
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 trend filter
    close_1w = df_1w['close'].values
    ema_34 = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # R3 = close + 1.1*(high - low)/2
    # S3 = close - 1.1*(high - low)/2
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate volume regime: current 12h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        close_val = close[i]
        ema_trend = ema_34_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(close_val) or np.isnan(ema_trend) or np.isnan(r3_level) or np.isnan(s3_level):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine regime: bull if close > 1w EMA34, bear if close < 1w EMA34
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Regime-based entry conditions
        if is_bull_regime:
            # Long: price breaks above R3 with volume spike
            long_entry = (close_val > r3_level) and vol_spike
        else:
            long_entry = False
            
        if is_bear_regime:
            # Short: price breaks below S3 with volume spike
            short_entry = (close_val < s3_level) and vol_spike
        else:
            short_entry = False
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit on price re-enters below R3 (loss of bullish momentum) or regime change to bear
            if close_val < r3_level or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on price re-enters above S3 (loss of bearish momentum) or regime change to bull
            if close_val > s3_level or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals