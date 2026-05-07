#!/usr/bin/env python3
name = "6h_ADX_Alligator_BullBear_Power"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA34 trend
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    trend_up = close > ema_34_1d_aligned
    trend_down = close < ema_34_1d_aligned
    
    # Calculate Alligator (SMMA) from 6h data
    # SMMA: Smoothed Moving Average (Jaw: 13-period, Teeth: 8-period, Lips: 5-period)
    # We'll use SMA as approximation for simplicity (SMMA requires initial SMA)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    # Calculate SMMA using cumulative smoothing (approximation)
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_smma * (period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, jaw_period)
    teeth = smma(close, teeth_period)
    lips = smma(close, lips_period)
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed TR, +DM, -DM
        def smma_series(arr, period):
            result = np.full_like(arr, np.nan)
            if len(arr) < period:
                return result
            result[period-1] = np.mean(arr[:period])
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
            return result
        
        tr_smoothed = smma_series(tr, period)
        plus_dm_smoothed = smma_series(plus_dm, period)
        minus_dm_smoothed = smma_series(minus_dm, period)
        
        # Avoid division by zero
        plus_di = np.where(tr_smoothed != 0, (plus_dm_smoothed / tr_smoothed) * 100, 0)
        minus_di = np.where(tr_smoothed != 0, (minus_dm_smoothed / tr_smoothed) * 100, 0)
        
        # DX and ADX
        dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
        adx = smma_series(dx, period)
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Elder Ray: Bull Power and Bear Power
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 2  # ~12 hours (2*6h) to prevent overtrading
    
    start_idx = max(jaw_period, teeth_period, lips_period, 14, 13)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(adx[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine Alligator alignment
        # Alligator sleeping: jaws, teeth, lips intertwined (no clear trend)
        # Alligator awake: jaws > teeth > lips (uptrend) or jaws < teeth < lips (downtrend)
        alligator_long = jaw[i] > teeth[i] > lips[i]
        alligator_short = jaw[i] < teeth[i] < lips[i]
        
        # Determine trend direction from 1d EMA34
        trending_up = trend_up[i]
        trending_down = trend_down[i]
        
        # Elder Ray conditions
        bull_power_positive = bull_power[i] > 0
        bear_power_negative = bear_power[i] < 0
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Alligator aligned up, ADX > 25 (strong trend), 1d uptrend, Bull Power positive
            if (alligator_long and adx[i] > 25 and trending_up and bull_power_positive):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Alligator aligned down, ADX > 25 (strong trend), 1d downtrend, Bear Power negative
            elif (alligator_short and adx[i] > 25 and trending_down and bear_power_negative):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Alligator alignment breaks or ADX weakens or 1d trend changes
            if not (alligator_long and adx[i] > 20 and trending_up and bull_power_positive):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Alligator alignment breaks or ADX weakens or 1d trend changes
            if not (alligator_short and adx[i] > 20 and trending_down and bear_power_negative):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Combines ADX trend strength, Alligator trend alignment, and Elder Ray power to capture strong trends in both bull and bear markets. ADX > 25 filters for strong trends, Alligator alignment confirms trend direction, and Elder Ray confirms bull/bear power. The 1d EMA34 trend filter ensures alignment with higher timeframe momentum. This combination reduces false signals and works in various market conditions. Target: 50-150 total trades over 4 years (12-37/year) to minimize fee disruption. Uses discrete position sizing (0.25) to balance risk and reward while reducing fee churn. This strategy is novel as it combines these three specific indicators in a 6h timeframe context, which hasn't been extensively tested in the provided experiments. The focus on trend strength and alignment should provide robust performance in both bull and bear markets for BTC and ETH.