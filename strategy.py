#!/usr/bin/env python3
"""
Experiment #399: 6h Elder Ray + 1d ADX Regime + Volume Confirmation

HYPOTHESIS: Elder Ray (Bull/Bear Power) identifies institutional buying/selling pressure.
Combined with 1d ADX regime filter (ADX>25 = trending, ADX<20 = ranging) and volume confirmation,
this strategy captures strong momentum moves while avoiding false signals in choppy markets.
Elder Ray works in both bull (strong Bull Power) and bear (strong Bear Power) markets.
Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_elder_ray_1d_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for ADX regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX(14) on 1d
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr = np.maximum(high_1d - low_1d, 
                        np.maximum(abs(high_1d - np.roll(close_1d, 1)), 
                                   abs(low_1d - np.roll(close_1d, 1))))
        tr[0] = high_1d[0] - low_1d[0]  # First TR
        
        # Directional Movement
        up_move = np.diff(high_1d, prepend=high_1d[0])
        down_move = np.diff(low_1d, prepend=low_1d[0]) * -1  # Positive values
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
        def wilders_smoothing(data, period):
            result = np.full_like(data, np.nan)
            if len(data) >= period:
                # First value is simple average
                result[period-1] = np.mean(data[:period])
                # Subsequent values: Wilder's smoothing
                for i in range(period, len(data)):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
            return result
        
        tr14 = wilders_smoothing(tr, 14)
        plus_dm14 = wilders_smoothing(plus_dm, 14)
        minus_dm14 = wilders_smoothing(minus_dm, 14)
        
        # DI+ and DI-
        plus_di14 = np.where(tr14 != 0, (plus_dm14 / tr14) * 100, 0)
        minus_di14 = np.where(tr14 != 0, (minus_dm14 / tr14) * 100, 0)
        
        # DX and ADX
        dx = np.where((plus_di14 + minus_di14) != 0, 
                      abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14) * 100, 0)
        adx = wilders_smoothing(dx, 14)
        
        # Align ADX to 6h timeframe
        adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    else:
        adx_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Elder Ray (Bull Power / Bear Power) ===
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    if n >= 13:
        close_s = pd.Series(close)
        ema13 = close_s.ewm(span=13, min_periods=13, adjust=False).mean().values
        bull_power = high - ema13
        bear_power = low - ema13
    else:
        ema13 = np.full(n, np.nan)
        bull_power = np.full(n, np.nan)
        bear_power = np.full(n, np.nan)
    
    # === 6h Indicators: Volume Spike (current vs 20-period average) ===
    if n >= 20:
        volume_s = pd.Series(volume)
        vol_ma20 = volume_s.rolling(window=20, min_periods=20).mean().values
        vol_ratio = np.zeros(n)
        vol_ratio[20:] = volume[20:] / vol_ma20[20:]
        vol_ratio[:20] = 1.0  # Neutral for warmup
    else:
        vol_ratio = np.full(n, 1.0)
    
    # === Session filter: 00-23 UTC (trade all hours for 6h timeframe) ===
    hours = prices.index.hour  # Pre-compute before loop
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Session Filter: Trade all hours for 6h timeframe ---
        hour = hours[i]
        # No session filter for 6h - trade continuously
        
        # --- Data Validity Check ---
        if (np.isnan(adx_aligned[i]) or np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit when Bull Power turns negative
                if bull_power[i] <= 0:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit when Bear Power turns positive
                if bear_power[i] >= 0:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Regime filter: ADX > 25 = trending market (good for Elder Ray signals)
        trending_market = adx_aligned[i] > 25
        
        # Volume confirmation: significant volume spike
        volume_confirmation = vol_ratio[i] > 1.5
        
        # Long: Strong Bull Power (buying pressure) + volume + trending market
        long_condition = (
            trending_market and
            volume_confirmation and
            bull_power[i] > 0 and  # Bull Power positive = buying pressure
            bull_power[i] > np.mean(bull_power[max(0, i-20):i]) if i >= 20 else bull_power[i] > 0  # Above recent average
        )
        
        # Short: Strong Bear Power (selling pressure) + volume + trending market
        short_condition = (
            trending_market and
            volume_confirmation and
            bear_power[i] < 0 and  # Bear Power negative = selling pressure
            bear_power[i] < np.mean(bear_power[max(0, i-20):i]) if i >= 20 else bear_power[i] < 0  # Below recent average
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals