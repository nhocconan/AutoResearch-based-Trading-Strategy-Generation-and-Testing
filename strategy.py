#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and ADX regime filter
# Long when price breaks above Camarilla R3 AND volume > 2.0x 20-bar avg AND 1d ADX > 25 (trending)
# Short when price breaks below Camarilla S3 AND volume > 2.0x 20-bar avg AND 1d ADX > 25 (trending)
# Exit when price crosses Camarilla R4/S4 levels (strong reversal) or ADX < 20 (range)
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 12-37 trades/year on 12h.
# Camarilla pivots provide mathematically derived support/resistance levels that work in ranging markets.
# Volume confirmation ensures breakouts have conviction, reducing false signals.
# 1d ADX filter ensures we only trade when higher timeframe is trending (avoids chop).
# Works in bull markets via upward breakouts, works in bear via downward breakouts with volume spikes.

name = "12h_Camarilla_R3S3_Breakout_1dADX_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for ADX(14)
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d.shift(1))).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d.shift(1))).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = pd.Series(low_1d).diff().abs()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    plus_di_1d = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_1d)
    minus_di_1d = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_1d)
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = dx_1d.ewm(alpha=1/14, adjust=False).mean()
    
    # Align 1d ADX to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d.values)
    
    # Calculate Camarilla levels on 12h data (based on previous bar)
    # Camarilla: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # Using previous bar's OHLC to avoid look-ahead
    camarilla_high = high
    camarilla_low = low
    camarilla_close = close
    
    # Shift by 1 to use previous bar for level calculation
    prev_high = np.roll(camarilla_high, 1)
    prev_low = np.roll(camarilla_low, 1)
    prev_close = np.roll(camarilla_close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # First bar invalid
    
    range_prev = prev_high - prev_low
    camarilla_r3 = prev_close + (range_prev * 1.1 / 4)
    camarilla_s3 = prev_close - (range_prev * 1.1 / 4)
    camarilla_r4 = prev_close + (range_prev * 1.1 / 2)
    camarilla_s4 = prev_close - (range_prev * 1.1 / 2)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Need sufficient history for ADX and volume
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(camarilla_r4[i]) or 
            np.isnan(camarilla_s4[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        adx_val = adx_1d_aligned[i]
        curr_close = close[i]
        prev_close_val = close[i-1]
        
        # Regime filter: only trade when ADX > 25 (trending)
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        # Handle exits first
        if position == 1:  # Long
            # Exit if price crosses below R4 (strong reversal) or ADX drops to ranging
            if curr_close < camarilla_r4[i] or is_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short
            # Exit if price crosses above S4 (strong reversal) or ADX drops to ranging
            if curr_close > camarilla_s4[i] or is_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        elif position == 0:  # Flat - look for new entries
            # Long when price breaks above R3 AND volume confirmation AND trending
            if curr_close > camarilla_r3[i] and prev_close_val <= camarilla_r3[i] and vol_conf and is_trending:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3 AND volume confirmation AND trending
            elif curr_close < camarilla_s3[i] and prev_close_val >= camarilla_s3[i] and vol_conf and is_trending:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals