#!/usr/bin/env python3
"""
Experiment #1955: 6h Elder Ray Index + 12h Regime + Volume Confirmation
HYPOTHESIS: Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) measures bull/bear strength. 
Combined with 12h ADX regime filter (ADX>25 = trend, ADX<20 = range) and volume confirmation, 
this captures strong directional moves while avoiding choppy markets. Works in both bull/bear regimes.
Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1955_6h_elder_ray_12h_regime_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for ADX regime filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ADX(14) for regime detection
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with original arrays
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
        def wilder_smooth(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            # First value is simple average
            if period < len(data):
                result[period-1] = np.nanmean(data[:period])
                for i in range(period, len(data)):
                    if not np.isnan(data[i]) and not np.isnan(result[i-1]):
                        result[i] = result[i-1] + alpha * (data[i] - result[i-1])
            return result
        
        atr = wilder_smooth(tr, period)
        dm_plus_smooth = wilder_smooth(dm_plus, period)
        dm_minus_smooth = wilder_smooth(dm_minus, period)
        
        # Avoid division by zero
        dm_plus_di = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
        dm_minus_di = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
        
        # DX and ADX
        dx = np.where((dm_plus_di + dm_minus_di) != 0, 
                      100 * np.abs(dm_plus_di - dm_minus_di) / (dm_plus_di + dm_minus_di), 0)
        adx = wilder_smooth(dx, period)
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    # Regime: ADX > 25 = trending, ADX < 20 = ranging
    regime_trending = np.where(adx_12h > 25, 1, 
                       np.where(adx_12h < 20, -1, 0))  # 1=trend, -1=range, 0=transition
    regime_trending_aligned = align_htf_to_ltf(prices, df_12h, regime_trending)
    
    # === 6h Indicators: EMA(13) for Elder Ray ===
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13  # negative values indicate bear strength
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for EMA(13), ADX, and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(regime_trending_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Exit if Bull Power turns negative (bulls losing strength)
                if bull_power[i] <= 0:
                    exit_signal = True
                # Exit if we enter ranging market (regime = -1)
                elif regime_trending_aligned[i] < 0:
                    exit_signal = True
            else:  # Short position
                # Exit if Bear Power turns positive (bears losing strength)
                if bear_power[i] >= 0:
                    exit_signal = True
                # Exit if we enter ranging market (regime = -1)
                elif regime_trending_aligned[i] < 0:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require trending regime (ADX > 25) for directional trades
        is_trending = regime_trending_aligned[i] > 0
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if is_trending and volume_spike:
            # Long entry: Bull Power positive AND rising (strong bullish momentum)
            if bull_power[i] > 0 and bull_power[i] > bull_power[i-1]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: Bear Power negative AND falling (strong bearish momentum)
            elif bear_power[i] < 0 and bear_power[i] < bear_power[i-1]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals