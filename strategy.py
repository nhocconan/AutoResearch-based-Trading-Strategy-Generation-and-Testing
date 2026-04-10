#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume confirmation and 1d ADX regime filter
# - Uses 4h Camarilla pivot levels (H3/L3) as intraday support/resistance
# - Long when price breaks above H3 with 4h volume > 1.5x 20-period average, short when breaks below L3
# - 1d ADX(14) > 25 filters for trending markets to avoid false breakouts in ranging conditions
# - Session filter: only trade 08:00-20:00 UTC to reduce noise
# - Discrete position sizing (0.20) minimizes fee churn
# - Target: 15-35 trades/year (60-140 total over 4 years) to stay within HARD MAX: 200 total
# - Camarilla H3/L3 levels provide tighter, more frequent breakouts than H4/L4 while maintaining reliability
# - Volume confirmation on 4h timeframe reduces false signals
# - Works in both bull and bear markets by following established trends with ADX filter

name = "1h_4h_1d_camarilla_breakout_volume_adx_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC for Camarilla pivots
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla pivot levels for 4h timeframe (H3/L3 for breakout)
    # Pivot = (High + Low + Close) / 3
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    # Range = High - Low
    range_4h = high_4h - low_4h
    # Camarilla levels: H3 = Close + Range * 1.1/4, L3 = Close - Range * 1.1/4
    camarilla_h3_4h = close_4h + range_4h * 1.1 / 4.0
    camarilla_l3_4h = close_4h - range_4h * 1.1 / 4.0
    
    # Align Camarilla levels to 1h timeframe (completed 4h bar only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3_4h)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3_4h)
    
    # Pre-compute 4h volume and its 20-period moving average for volume confirmation
    volume_4h = df_4h['volume'].values
    volume_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_20_4h)
    
    # Pre-compute 1d ADX(14) for regime filter (trending market detection)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum.reduce([tr1, tr2, tr3])
    
    # +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = np.nan
    down_move[0] = np.nan
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    tr_period = 14
    atr_1d = pd.Series(tr).rolling(window=tr_period, min_periods=tr_period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=tr_period, min_periods=tr_period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=tr_period, min_periods=tr_period).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / atr_1d
    minus_di = 100 * minus_dm_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).rolling(window=tr_period, min_periods=tr_period).mean().values
    
    # Align ADX to 1h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Pre-compute 1h ATR for risk management
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    
    tr1_1h = high_1h - low_1h
    tr2_1h = np.abs(high_1h - np.roll(close_1h, 1))
    tr3_1h = np.abs(low_1h - np.roll(close_1h, 1))
    tr1_1h[0] = np.nan
    tr2_1h[0] = np.nan
    tr3_1h[0] = np.nan
    tr_1h = np.maximum.reduce([tr1_1h, tr2_1h, tr3_1h])
    atr_1h = pd.Series(tr_1h).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute session filter (08:00-20:00 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Session filter: only trade 08:00-20:00 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(atr_1h[i]) or not in_session):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Get current 4h volume for filter (use raw volume, aligned)
        volume_4h_current = volume_4h
        volume_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h_current)
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        volume_confirm = volume_4h_aligned[i] > 1.5 * volume_ma_aligned[i]
        
        # Regime filter: ADX > 25 indicates trending market
        trending_market = adx_aligned[i] > 25.0
        
        close_price = close_1h[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Camarilla H3 AND volume confirmation AND trending market
            if close_price > camarilla_h3_aligned[i] and volume_confirm and trending_market:
                position = 1
                signals[i] = 0.20
            # Short conditions: price breaks below Camarilla L3 AND volume confirmation AND trending market
            elif close_price < camarilla_l3_aligned[i] and volume_confirm and trending_market:
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit conditions
            # Exit on opposite Camarilla level break or loss of momentum
            if position == 1:  # Long position
                # Exit if price breaks below Camarilla L3 (reversal signal)
                if close_price < camarilla_l3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            else:  # position == -1, Short position
                # Exit if price breaks above Camarilla H3 (reversal signal)
                if close_price > camarilla_h3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
    
    return signals