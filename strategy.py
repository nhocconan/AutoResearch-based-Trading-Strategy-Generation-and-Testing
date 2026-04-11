#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume confirmation and 1d ADX trend filter
# - Long: price breaks above H4 Camarilla level, 4h volume > 1.5x 20-period avg, 1d ADX(14) > 25 (trending)
# - Short: price breaks below L4 Camarilla level, 4h volume > 1.5x 20-period avg, 1d ADX(14) > 25 (trending)
# - Exit: price returns to H3/L3 level or ATR-based stop
# - Uses discrete position sizing: ±0.20 to limit drawdown and reduce fee churn
# - Target: 15-35 trades/year (60-140 total over 4 years) to stay within fee drag limits
# - Camarilla pivots work well in ranging markets; combined with volume/trend filters to capture breakouts in both bull/bear markets

name = "1h_4h_1d_camarilla_adx_volume_v1"
timeframe = "1h"
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
    
    # Load 4h data ONCE before loop for volume confirmation (MTF rule compliance)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return signals
    
    # Load 1d data ONCE before loop for ADX trend filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    tr_14 = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 1h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute 4h volume confirmation (20-period average)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # True Range for 4h
    tr_4h = np.maximum(high_4h - low_4h, np.maximum(np.abs(high_4h - np.roll(close_4h, 1)), np.abs(low_4h - np.roll(close_4h, 1))))
    tr_4h[0] = high_4h[0] - low_4h[0]
    
    # 4h ATR for Camarilla calculation
    atr_4h = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate Camarilla levels for 4h (based on previous 4h bar)
    # H4 = close + 1.1/2 * (high - low)
    # L4 = close - 1.1/2 * (high - low)
    # H3 = close + 1.1/4 * (high - low)
    # L3 = close - 1.1/4 * (high - low)
    camarilla_high = close_4h + 1.1/2 * (high_4h - low_4h)
    camarilla_low = close_4h - 1.1/2 * (high_4h - low_4h)
    camarilla_h3 = close_4h + 1.1/4 * (high_4h - low_4h)
    camarilla_l3 = close_4h - 1.1/4 * (high_4h - low_4h)
    
    # Align Camarilla levels to 1h timeframe
    camarilla_high_aligned = align_htf_to_ltf(prices, df_4h, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_4h, camarilla_low)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # Pre-compute 4h volume confirmation (20-period average)
    volume_sma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_sma_20_4h)
    
    # Pre-compute ATR for regime detection and stoploss (1h)
    tr_1h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_1h[0] = high[0] - low[0]
    atr_14 = pd.Series(tr_1h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_sma_20_4h_aligned[i]) or np.isnan(atr_14[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade between 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels
        h4 = camarilla_high_aligned[i]
        l4 = camarilla_low_aligned[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        # We need to get the current 4h volume bar that aligns with this 1h bar
        # Since we aligned the 4h volume SMA, we check if current 1h volume contributes to a 4h bar that meets criteria
        vol_confirm = volume_current > 1.5 * volume_sma_20_4h_aligned[i] / 4  # Approximate per 1h bar
        
        # Trend filter: 1d ADX > 25 (indicates trending market)
        adx_trend = adx_aligned[i] > 25
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above H4 Camarilla level, volume confirmation, trending
        if close_price > h4 and vol_confirm and adx_trend:
            enter_long = True
        
        # Short breakout: price below L4 Camarilla level, volume confirmation, trending
        if close_price < l4 and vol_confirm and adx_trend:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to H3 level or ATR-based stop
            exit_long = (close_price <= h3) or (close_price <= entry_price - 1.5 * atr_14[i])
        elif position == -1:
            # Exit short if price returns to L3 level or ATR-based stop
            exit_short = (close_price >= l3) or (close_price >= entry_price + 1.5 * atr_14[i])
        
        # Track entry price for stoploss calculation
        if enter_long or enter_short:
            entry_price = close_price
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.20
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals