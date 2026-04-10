#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + ADX Regime
# - Bull Power = High - EMA13, Bear Power = EMA13 - Low (on 6h)
# - Regime filter: ADX(14) > 25 for trending, < 20 for ranging (with hysteresis)
# - In trending regime (ADX > 25): Long when Bull Power > 0 and rising, Short when Bear Power > 0 and rising
# - In ranging regime (ADX < 20): Mean reversion at Bollinger Bands (20,2) - Long at lower band, Short at upper band
# - Uses 1w trend filter: only take longs when price > weekly EMA50, shorts when price < weekly EMA50
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years)

name = "6h_elder_ray_adx_regime_1wfilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w indicators (trend filter)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute 1d indicators for ADX and Bollinger Bands
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX calculation (14-period)
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    plus_di_1d = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Bollinger Bands (20,2) on 1d
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + 2 * std_20_1d
    lower_bb_1d = sma_20_1d - 2 * std_20_1d
    upper_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d)
    lower_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d)
    
    # Pre-compute 6h indicators for Elder Ray
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # EMA(13) for Elder Ray
    ema_13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high_6h - ema_13_6h
    bear_power = ema_13_6h - low_6h
    
    # Signals array
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Hysteresis thresholds for ADX regime
    adx_trend_enter = 25   # Enter trending regime when ADX > 25
    adx_range_enter = 20   # Enter ranging regime when ADX < 20
    adx_trend_exit = 18    # Exit trending regime when ADX < 18
    adx_range_exit = 22    # Exit ranging regime when ADX > 22
    
    in_trending_regime = False
    in_ranging_regime = False
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(upper_bb_1d_aligned[i]) or np.isnan(lower_bb_1d_aligned[i]) or
            np.isnan(ema_13_6h[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Determine regime with hysteresis
        adx_val = adx_1d_aligned[i]
        if not in_trending_regime and not in_ranging_regime:
            # Initially determine regime
            if adx_val > adx_trend_enter:
                in_trending_regime = True
            elif adx_val < adx_range_enter:
                in_ranging_regime = True
        elif in_trending_regime:
            if adx_val < adx_trend_exit:
                in_trending_regime = False
                in_ranging_regime = (adx_val < adx_range_enter)
        elif in_ranging_regime:
            if adx_val > adx_range_exit:
                in_ranging_regime = False
                in_trending_regime = (adx_val > adx_trend_enter)
        
        # Regime-based logic
        if in_trending_regime:
            # Trending regime: Elder Ray momentum
            # Long when Bull Power > 0 and rising (previous bar Bull Power < current)
            # Short when Bear Power > 0 and rising (previous bar Bear Power < current)
            if i > 0:
                bull_power_rising = bull_power[i] > bull_power[i-1]
                bear_power_rising = bear_power[i] > bear_power[i-1]
                
                # Long condition: Bull Power positive and rising, price above weekly EMA50
                if (bull_power[i] > 0 and bull_power_rising and 
                    close_6h[i] > ema_50_1w_aligned[i]):
                    if position != 1:
                        position = 1
                        signals[i] = 0.25
                    else:
                        signals[i] = 0.25
                # Short condition: Bear Power positive and rising, price below weekly EMA50
                elif (bear_power[i] > 0 and bear_power_rising and 
                      close_6h[i] < ema_50_1w_aligned[i]):
                    if position != -1:
                        position = -1
                        signals[i] = -0.25
                    else:
                        signals[i] = -0.25
                else:
                    # Hold current position or flat
                    signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            else:
                signals[i] = 0.0
                
        elif in_ranging_regime:
            # Ranging regime: Bollinger Band mean reversion
            # Long at lower band, Short at upper band
            if close_6h[i] <= lower_bb_1d_aligned[i]:
                if position != 1:
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            elif close_6h[i] >= upper_bb_1d_aligned[i]:
                if position != -1:
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:
                # Hold current position or flat
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            # Undefined regime - stay flat
            signals[i] = 0.0
            position = 0
    
    return signals