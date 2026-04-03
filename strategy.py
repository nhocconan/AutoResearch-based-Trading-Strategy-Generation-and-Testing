#!/usr/bin/env python3
"""
Experiment #1599: 6h Elder Ray + ADX Regime + Volume Confirmation
HYPOTHESIS: Elder Ray (Bull/Bear Power) combined with ADX regime filter and volume confirmation captures sustainable moves in both bull and bear markets. Bull Power > 0 + Bear Power < 0 indicates balanced momentum; ADX > 25 filters for trending conditions; volume > 1.2x average confirms participation. 6h timeframe reduces noise while allowing sufficient trades. Position size 0.25 balances return and drawdown. Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1599_6h_elder_ray_adx_regime_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # === HTF: 12h data for ADX regime filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # ADX(14) calculation on 12h
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        tr = np.zeros(len(high))
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        # Smooth using Wilder's smoothing (alpha = 1/period)
        alpha = 1.0 / period
        plus_dm_smooth = np.zeros_like(plus_dm)
        minus_dm_smooth = np.zeros_like(minus_dm)
        tr_smooth = np.zeros_like(tr)
        plus_dm_smooth[period] = plus_dm[1:period+1].sum()
        minus_dm_smooth[period] = minus_dm[1:period+1].sum()
        tr_smooth[period] = tr[1:period+1].sum()
        for i in range(period+1, len(high)):
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / period) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / period) + minus_dm[i]
            tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / period) + tr[i]
        plus_di = 100 * plus_dm_smooth / (tr_smooth + 1e-10)
        minus_di = 100 * minus_dm_smooth / (tr_smooth + 1e-10)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = np.zeros_like(dx)
        adx[2*period] = dx[period+1:2*period+1].mean()
        for i in range(2*period+1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # === 6h Indicators: Elder Ray (Bull/Bear Power) ===
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
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
    
    warmup = max(20, 13)  # sufficient for EMA and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx_12h_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic: Reverse on signal change ---
        if in_position:
            # Check for exit conditions
            long_exit = (bull_power[i] <= 0) or (adx_12h_aligned[i] < 20) or (vol_ratio[i] < 1.0)
            short_exit = (bear_power[i] >= 0) or (adx_12h_aligned[i] < 20) or (vol_ratio[i] < 1.0)
            
            if (position_side > 0 and long_exit) or (position_side < 0 and short_exit):
                in_position = False
                position_side = 0
                signals[i] = 0.0
                continue
        
        # --- New Position Entry Logic ---
        # Require ADX > 25 for trending regime
        trending = adx_12h_aligned[i] > 25
        
        # Volume confirmation: require volume > 1.2x average
        volume_confirm = vol_ratio[i] > 1.2
        
        if trending and volume_confirm:
            # Elder Ray signals: Bull Power > 0 AND Bear Power < 0 for balanced momentum
            # Actually, we want divergence: Bull Power rising while prices fall (bullish) OR Bear Power falling while prices rise (bearish)
            # Simplified: Bull Power > 0 for long, Bear Power < 0 for short
            if bull_power[i] > 0 and bear_power[i] < 0:
                # Both positive and negative power - wait for confirmation
                # Use price close relative to open for direction
                if close[i] > prices["open"].iloc[i]:  # bullish candle
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    signals[i] = SIZE
                elif close[i] < prices["open"].iloc[i]:  # bearish candle
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            elif bull_power[i] > 0:  # Strong bullish pressure
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            elif bear_power[i] < 0:  # Strong bearish pressure
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals