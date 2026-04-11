#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot + 1d volume spike + ADX trend filter
# - Camarilla levels from 1d: key support/resistance levels for mean reversion
# - Long when price touches S3 level with volume > 2.5x 20-period average (strong reversal)
# - Short when price touches R3 level with volume > 2.5x 20-period average
# - ADX trend filter: only trade when ADX < 25 to avoid strong trends and focus on mean reversion
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits for 12h
# - Works in both bull (reversals at support) and bear (reversals at resistance) markets
# - 1d HTF provides reliable Camarilla levels and volume confirmation

name = "12h_1d_camarilla_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Camarilla levels, volume, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Extract 1d arrays
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels for each 1d bar
    # Camarilla: pivot = (H+L+C)/3, range = H-L
    # S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4
    # R1 = C + (H-L)*1.1/12, R2 = C + (H-L)*1.1/6, R3 = C + (H-L)*1.1/4
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    s3_1d = close_1d - (range_1d * 1.1 / 4.0)  # Strongest support
    r3_1d = close_1d + (range_1d * 1.1 / 4.0)  # Strongest resistance
    
    # 1d volume SMA (20-period)
    volume_series = pd.Series(volume_1d)
    volume_sma_20_1d = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # 1d ADX (14-period) - trend strength filter
    # True Range
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = pd.Series(high_1d).diff()
    dm_minus = -pd.Series(low_1d).diff()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smoothed values
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr_14_1d
    di_minus = 100 * dm_minus_smooth / atr_14_1d
    
    # DX and ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_14_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 12h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Camarilla touch conditions (with small tolerance for wicks)
        tolerance = 0.001  # 0.1% tolerance for level touch
        touch_s3 = abs(price_low - s3_aligned[i]) / s3_aligned[i] <= tolerance
        touch_r3 = abs(price_high - r3_aligned[i]) / r3_aligned[i] <= tolerance
        
        # Volume confirmation: current volume > 2.5x 20-period average
        vol_confirm = volume_current > 2.5 * volume_sma_20_aligned[i]
        
        # ADX trend filter: only trade when ADX < 25 (range-bound market)
        adx_filter = adx_aligned[i] < 25.0
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price touches S3 support + volume confirmation + ADX filter (range market)
        if touch_s3 and vol_confirm and adx_filter:
            enter_long = True
        
        # Short: Price touches R3 resistance + volume confirmation + ADX filter
        if touch_r3 and vol_confirm and adx_filter:
            enter_short = True
        
        # Exit conditions: opposite touch or volatility breakout
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price touches R3 resistance or ADX indicates strong trend
            exit_long = touch_r3 or (adx_aligned[i] >= 30.0)
        elif position == -1:
            # Exit short if price touches S3 support or ADX indicates strong trend
            exit_short = touch_s3 or (adx_aligned[i] >= 30.0)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals