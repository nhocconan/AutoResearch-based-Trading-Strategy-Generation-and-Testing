#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull Power/Bear Power) with 12h ADX regime filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# 12h ADX > 25 indicates trending market (use Elder Ray signals), ADX < 20 indicates ranging (fade extremes)
# Volume confirmation (>1.5x 20-period average) ensures institutional participation
# Works in both bull/bear markets by adapting to regime: trend follow in trending, mean revert in ranging
# Discrete position sizing (0.25) minimizes fee churn while maintaining meaningful exposure.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.

name = "6h_ElderRay_12hADX_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Calculate 12h EMA14 and ADX components for trend regime
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # EMA14 for 12h timeframe
    ema_14_12h = pd.Series(close_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ADX calculation (14-period)
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with original length
    
    # Directional Movement
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di_12h = 100 * plus_dm_14 / tr_14
    minus_di_12h = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate 20-period average volume for spike confirmation (on 6h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # ADX and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema13 = ema_13[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_adx = adx_12h_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = curr_volume > 1.5 * curr_vol_ma
        
        # Regime determination
        is_trending = curr_adx > 25
        is_ranging = curr_adx < 20
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: bear power turns negative OR trend weakens
            if curr_bear_power > 0 or (is_trending and curr_adx < 22):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bull power turns positive OR trend weakens
            if curr_bull_power < 0 or (is_trending and curr_adx < 22):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            if is_trending:
                # Trending regime: follow Elder Ray momentum
                # Long: strong bull power AND volume confirmation
                if curr_bull_power > 0 and vol_confirm:
                    signals[i] = 0.25
                    position = 1
                # Short: strong bear power AND volume confirmation
                elif curr_bear_power < 0 and vol_confirm:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif is_ranging:
                # Ranging regime: mean reversion at extremes
                # Long: bear power negative (oversold) AND volume confirmation
                if curr_bear_power < 0 and vol_confirm:
                    signals[i] = 0.25
                    position = 1
                # Short: bull power positive (overbought) AND volume confirmation
                elif curr_bull_power > 0 and vol_confirm:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Transition regime: no trades
                signals[i] = 0.0
    
    return signals