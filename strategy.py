#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h volume confirmation and 1d ADX trend filter
# - Long: price breaks above Camarilla R3 level (1d), volume > 1.3x 20-period avg (12h), 1d ADX(14) > 20
# - Short: price breaks below Camarilla S3 level (1d), volume > 1.3x 20-period avg (12h), 1d ADX(14) > 20
# - Exit: price returns to Camarilla pivot point (PP) or ATR-based stop (2.0x ATR)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-25 trades/year (50-100 total over 4 years) to stay within fee drag limits
# - Camarilla pivots work well in ranging markets; volume and ADX filter ensure breakout validity

name = "6h_12h_1d_camarilla_adx_volume_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load 1d data ONCE before loop for Camarilla pivot and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d Camarilla pivots (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Camarilla levels: based on previous day's range
    # PP = (H + L + C) / 3
    # R3 = PP + (H - L) * 1.1 / 2
    # S3 = PP - (H - L) * 1.1 / 2
    # R4 = PP + (H - L) * 1.1
    # S4 = PP - (H - L) * 1.1
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_open = np.roll(open_1d, 1)
    
    # First bar: use same values (no look-ahead)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    prev_open[0] = open_1d[0]
    
    pp = (prev_high + prev_low + prev_close) / 3.0
    camarilla_range = prev_high - prev_low
    r3 = pp + camarilla_range * 1.1 / 2.0
    s3 = pp - camarilla_range * 1.1 / 2.0
    r4 = pp + camarilla_range * 1.1
    s4 = pp - camarilla_range * 1.1
    camarilla_pp = pp  # pivot point for exit
    
    # Pre-compute 1d ADX(14) for trend filter
    # True Range
    tr_1d = np.maximum(prev_high - prev_low, np.maximum(np.abs(prev_high - prev_close), np.abs(prev_low - prev_close)))
    tr_1d[0] = prev_high[0] - prev_low[0]
    
    # Directional Movement
    dm_plus = np.where((prev_high - np.roll(prev_high, 1)) > (np.roll(prev_low, 1) - prev_low), np.maximum(prev_high - np.roll(prev_high, 1), 0), 0)
    dm_minus = np.where((np.roll(prev_low, 1) - prev_low) > (prev_high - np.roll(prev_high, 1)), np.maximum(np.roll(prev_low, 1) - prev_low, 0), 0)
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
    
    # Load 12h data ONCE before loop for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return signals
    
    volume_12h = df_12h['volume'].values
    volume_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d Camarilla and ADX to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Align 12h volume SMA to 6h timeframe
    volume_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    
    # Pre-compute 6h ATR for stoploss
    tr_6h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_6h[0] = high[0] - low[0]
    atr_14_6h = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(camarilla_pp_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_sma_20_12h_aligned[i]) or np.isnan(atr_14_6h[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        pp_level = camarilla_pp_aligned[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average (12h)
        vol_confirm = volume_current > 1.3 * volume_sma_20_12h_aligned[i]
        
        # Trend filter: 1d ADX > 20 (indicates trending market)
        adx_trend = adx_aligned[i] > 20
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above Camarilla R3, volume confirmation, trending
        if close_price > r3_level and vol_confirm and adx_trend:
            enter_long = True
        
        # Short breakout: price below Camarilla S3, volume confirmation, trending
        if close_price < s3_level and vol_confirm and adx_trend:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to pivot point or ATR-based stop
            exit_long = (close_price <= pp_level) or (close_price <= entry_price - 2.0 * atr_14_6h[i])
        elif position == -1:
            # Exit short if price returns to pivot point or ATR-based stop
            exit_short = (close_price >= pp_level) or (close_price >= entry_price + 2.0 * atr_14_6h[i])
        
        # Track entry price for stoploss calculation
        if enter_long or enter_short:
            entry_price = close_price
        
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