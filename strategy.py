#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray with 12h regime filter
# Uses Bull Power (EMA13 - Low) and Bear Power (High - EMA13) to measure bull/bear strength
# Regime filter: 12h ADX > 25 for trending markets, < 20 for ranging
# In trending regime (ADX > 25): trend follow with Elder Ray
# In ranging regime (ADX < 20): mean revert at extremes
# Volume confirmation required for all entries
# Target: 50-150 total trades over 4 years with controlled risk in both bull and bear markets

name = "6h_elderray_12h_regime_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for regime filter (ADX)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed values
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(span=period, adjust=False).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(span=period, adjust=False).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
        di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # 6h data for Elder Ray
    # EMA13 for Elder Ray calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    
    # Bull Power = EMA13 - Low
    bull_power = ema13 - low
    # Bear Power = High - EMA13
    bear_power = high - ema13
    
    # Volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Regime determination
        adx_val = adx_12h_aligned[i]
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR approximation
            atr_approx = np.abs(high[i] - low[i])
            if close[i] < entry_price - 2.0 * atr_approx:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit conditions based on regime
            elif is_trending and bull_power[i] < 0:  # trend weakening
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            elif is_ranging and bear_power[i] > 0:  # mean revert signal
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR approximation
            atr_approx = np.abs(high[i] - low[i])
            if close[i] > entry_price + 2.0 * atr_approx:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit conditions based on regime
            elif is_trending and bear_power[i] < 0:  # trend weakening
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            elif is_ranging and bull_power[i] < 0:  # mean revert signal
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            vol_confirm = volume[i] > 1.5 * volume_ma[i]
            
            if is_trending:
                # Trend following: Elder Ray signals
                if bull_power[i] > 0 and bear_power[i] < 0 and vol_confirm:  # strong bull
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif bear_power[i] > 0 and bull_power[i] < 0 and vol_confirm:  # strong bear
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            elif is_ranging:
                # Mean reversion: fade extremes
                if bull_power[i] < -0.5 * np.std(bull_power[max(0, i-50):i+1]) and vol_confirm:  # oversold
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif bear_power[i] < -0.5 * np.std(bear_power[max(0, i-50):i+1]) and vol_confirm:  # overbought
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals