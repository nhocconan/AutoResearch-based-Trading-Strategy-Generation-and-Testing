#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d regime detection (ADX for trend, Bollinger BW for range) + 4h Donchian breakout
# In trending regime (ADX > 25): follow 4h Donchian breakout
# In ranging regime (ADX < 20): mean-revert at Bollinger Bands (20,2) on 1h
# Volume confirmation on both modes
# Session filter 08-20 UTC to reduce noise
# Position size: 0.20 (discrete to minimize churn)
# Target: 20-50 trades/year (80-200 over 4 years) to avoid fee drag

name = "1h_4h1d_ADXBB_Donchian_Regime_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend detection and Donchian
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h ADX for trend detection
    tr1 = df_4h['high'] - df_4h['low']
    tr2 = abs(df_4h['high'] - df_4h['close'].shift(1))
    tr3 = abs(df_4h['low'] - df_4h['close'].shift(1))
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h = tr_4h.rolling(window=14, min_periods=14).mean()
    up_move = df_4h['high'] - df_4h['high'].shift(1)
    down_move = df_4h['low'].shift(1) - df_4h['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr_4h
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr_4h
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_4h = dx.rolling(window=14, min_periods=14).mean()
    
    # 4h Donchian channels (20-period)
    donchian_high_4h = df_4h['high'].rolling(window=20, min_periods=20).max()
    donchian_low_4h = df_4h['low'].rolling(window=20, min_periods=20).min()
    
    # Get 1h Bollinger Bands for mean reversion
    close_1h = pd.Series(close)
    sma_20 = close_1h.rolling(window=20, min_periods=20).mean()
    std_20 = close_1h.rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Get 1d volume for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    vol_ma_20_1d = df_1d['volume'].rolling(window=20, min_periods=20).mean()
    vol_ratio_1d = df_1d['volume'] / vol_ma_20_1d
    
    # Align all HTF/indicators to 1h
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h.values)
    donchian_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h.values)
    donchian_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h.values)
    upper_bb_aligned = align_htf_to_ltf(prices, df_4h, upper_bb.values)  # using 4h as proxy for alignment timing
    lower_bb_aligned = align_htf_to_ltf(prices, df_4h, lower_bb.values)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d.values, additional_delay_bars=1)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # warmup for indicators
        # Skip if any critical value is NaN or outside session
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(donchian_high_4h_aligned[i]) or 
            np.isnan(donchian_low_4h_aligned[i]) or np.isnan(upper_bb_aligned[i]) or
            np.isnan(lower_bb_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime detection
        is_trending = adx_4h_aligned[i] > 25
        is_ranging = adx_4h_aligned[i] < 20
        
        # Volume confirmation (1d)
        vol_confirm = vol_ratio_1d_aligned[i] > 1.5
        
        if position == 0:
            if is_trending and vol_confirm:
                # Trend following: 4h Donchian breakout
                if close[i] > donchian_high_4h_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                elif close[i] < donchian_low_4h_aligned[i]:
                    signals[i] = -0.20
                    position = -1
            elif is_ranging and vol_confirm:
                # Mean reversion: 1h Bollinger Bands
                if close[i] < lower_bb_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                elif close[i] > upper_bb_aligned[i]:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Exit conditions
            if is_trending:
                # Exit trend long on Donchian breakdown
                if close[i] < donchian_low_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # ranging
                # Exit mean reversion at mean
                if close[i] > sma_20.iloc[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
        elif position == -1:
            # Exit conditions
            if is_trending:
                # Exit trend short on Donchian breakout
                if close[i] > donchian_high_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
            else:  # ranging
                # Exit mean reversion at mean
                if close[i] < sma_20.iloc[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals