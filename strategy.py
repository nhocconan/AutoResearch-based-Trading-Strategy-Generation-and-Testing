#!/usr/bin/env python3
"""
Experiment #2819: 6h Elder Ray Power + 12h ADX Regime + Volume Spike
HYPOTHESIS: Combines Elder Ray (bull/bear power) for momentum strength with 12h ADX regime filter 
(trending vs ranging) and volume confirmation to capture strong directional moves while avoiding 
whipsaws in low-volume or choppy markets. Works in both bull/bear markets by adapting to regime.
Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2819_6h_elder_ray_adx_regime_vol_v1"
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
    
    # Calculate ADX(14) on 12h
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        atr[period] = np.nansum(tr[1:period+1]) / period
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        if np.sum(atr[period+1:]) > 0:
            plus_sm = np.zeros_like(high)
            minus_sm = np.zeros_like(high)
            plus_sm[period] = np.nansum(plus_dm[1:period+1]) / period
            minus_sm[period] = np.nansum(minus_dm[1:period+1]) / period
            
            for i in range(period+1, len(high)):
                plus_sm[i] = (plus_sm[i-1] * (period-1) + plus_dm[i]) / period
                minus_sm[i] = (minus_sm[i-1] * (period-1) + minus_dm[i]) / period
                
                plus_di[i] = 100 * plus_sm[i] / atr[i]
                minus_di[i] = 100 * minus_sm[i] / atr[i]
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.full_like(high, np.nan)
        if np.sum(dx[2*period+1:]) > 0:
            adx[2*period] = np.nansum(dx[period+1:2*period+1]) / period
            for i in range(2*period+1, len(high)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # === 6h Indicators: Elder Ray Power, EMA(13), Volume MA(20) ===
    # EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray Power: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume MA for spike detection
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
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(adx_12h_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if Elder Ray turns bearish OR ADX drops below 20 (regime change to ranging)
                if bear_power[i] > bull_power[i] or adx_12h_aligned[i] < 20:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Trailing stop: exit if price drops 2.5*ATR below highest since entry
                # ATR approximation using 6h range
                atr_estimate = (high[i] - low[i]) * 0.15  # approximate ATR
                if price < highest_since_entry - 2.5 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if Elder Ray turns bullish OR ADX drops below 20 (regime change to ranging)
                if bull_power[i] > bear_power[i] or adx_12h_aligned[i] < 20:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Trailing stop: exit if price rises 2.5*ATR above lowest since entry
                atr_estimate = (high[i] - low[i]) * 0.15
                if price > lowest_since_entry + 2.5 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Regime filter: require ADX > 25 for trending market
        trending_regime = adx_12h_aligned[i] > 25
        
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        if trending_regime and volume_spike:
            # Long entry: Bull Power positive and increasing (momentum building)
            if bull_power[i] > 0 and bull_power[i] > bull_power[i-1]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: Bear Power negative and decreasing (momentum building)
            elif bear_power[i] < 0 and bear_power[i] < bear_power[i-1]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals