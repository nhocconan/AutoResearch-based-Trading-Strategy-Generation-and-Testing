#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Camarilla pivot levels for mean reversion and 1w ADX for trend strength
# - Uses 1d HTF to calculate Camarilla pivot levels (R3, R4, S3, S4) from prior day
# - Uses 1w HTF for ADX: ADX > 25 indicates strong trend (avoid mean reversion in strong trends)
# - In ranging markets (ADX <= 25): fade at R3/S3 levels (sell at R3, buy at S3)
# - In trending markets (ADX > 25): breakout continuation at R4/S4 levels (buy at R4, sell at S4)
# - Volume confirmation: current 6h volume > 1.2x 20-period average to filter low-quality signals
# - Fixed position size 0.25 to control drawdown
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1d_1w_camarilla_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Camarilla pivot levels (based on prior day's OHLC)
    # Pivot = (High + Low + Close) / 3
    # Range = High - Low
    # R3 = Close + Range * 1.1/2
    # R4 = Close + Range * 1.1
    # S3 = Close - Range * 1.1/2
    # S4 = Close - Range * 1.1
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels for current day (based on prior day's data)
    camarilla_pivot = typical_price_1d
    camarilla_r3 = close_1d + range_1d * 1.1 / 2
    camarilla_r4 = close_1d + range_1d * 1.1
    camarilla_s3 = close_1d - range_1d * 1.1 / 2
    camarilla_s4 = close_1d - range_1d * 1.1
    
    # Calculate 1w ADX (14 periods)
    # +DM = max(High - Previous High, 0) if High - Previous High > Previous Low - Low else 0
    # -DM = max(Previous Low - Low, 0) if Previous Low - Low > High - Previous High else 0
    # TR = max(High - Low, abs(High - Previous Close), abs(Low - Previous Close))
    # +DI = 100 * EWMA(+DM) / ATR
    # -DI = 100 * EWMA(-DM) / ATR
    # ADX = EWMA(abs(+DI - -DI) / (+DI + -DI))
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = tr2[0] = tr3[0] = 0  # First period has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    up_move[0] = down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align all HTF data to 6h timeframe (wait for completed HTF bar)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_pivot_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.2x average
        volume_confirmed = volume[i] > 1.2 * vol_ma_20[i]
        
        # Determine market regime based on ADX
        ranging_market = adx_aligned[i] <= 25
        trending_market = adx_aligned[i] > 25
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit conditions
            if ranging_market:
                # In ranging market: exit when price reaches pivot or S3/S4
                if close[i] <= camarilla_pivot_aligned[i] or close[i] >= camarilla_r3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            else:  # trending_market
                # In trending market: exit when price reaches S4 or reverses below R3
                if close[i] <= camarilla_s4_aligned[i] or close[i] < camarilla_r3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
                    
        elif position == -1:  # Short position
            # Exit conditions
            if ranging_market:
                # In ranging market: exit when price reaches pivot or R3/R4
                if close[i] >= camarilla_pivot_aligned[i] or close[i] <= camarilla_s3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
            else:  # trending_market
                # In trending market: exit when price reaches R4 or reverses above S3
                if close[i] >= camarilla_r4_aligned[i] or close[i] > camarilla_s3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
        else:  # Flat
            # Entry logic based on market regime and Camarilla levels
            if volume_confirmed:
                if ranging_market:
                    # In ranging market: fade at R3/S3
                    if close[i] <= camarilla_s3_aligned[i] and close[i] > camarilla_s4_aligned[i]:
                        # Near S3, expect bounce to pivot - long
                        position = 1
                        signals[i] = position_size
                    elif close[i] >= camarilla_r3_aligned[i] and close[i] < camarilla_r4_aligned[i]:
                        # Near R3, expect pullback to pivot - short
                        position = -1
                        signals[i] = -position_size
                else:  # trending_market
                    # In trending market: breakout continuation at R4/S4
                    if close[i] >= camarilla_r4_aligned[i]:
                        # Break above R4, expect continuation - long
                        position = 1
                        signals[i] = position_size
                    elif close[i] <= camarilla_s4_aligned[i]:
                        # Break below S4, expect continuation - short
                        position = -1
                        signals[i] = -position_size
    
    return signals