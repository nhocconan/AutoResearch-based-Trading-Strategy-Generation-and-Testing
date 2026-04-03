#!/usr/bin/env python3
"""
Experiment #1631: 6h Elder Ray + 1d Regime Filter (ADX/Chop) + Volume Spike
HYPOTHESIS: Elder Ray (Bull Power/Bear Power) identifies institutional buying/selling pressure. 
Combined with 1d regime filter (ADX>25 AND Chop<61.8 for trending, Chop>=61.8 for ranging) 
and volume confirmation (>1.5x average), this strategy captures strong moves in both bull 
and bear markets while avoiding chop. Uses discrete position sizing (0.25) to limit drawdown 
during 2022 crash. Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1631_6h_elder_ray_1d_regime_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 1d Indicators: ADX(14) and Choppiness Index(14) ===
    # True Range
    tr_1d = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        tr_1d[i] = max(high_1d[i] - low_1d[i], 
                       abs(high_1d[i] - close_1d[i-1]), 
                       abs(low_1d[i] - close_1d[i-1]))
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    # +DM and -DM
    plus_dm_1d = np.zeros(len(close_1d))
    minus_dm_1d = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        plus_dm_1d[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm_1d[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # Smoothed values
    def rma(arr, period):
        """Wilder's smoothing (EMA with alpha=1/period)"""
        if len(arr) < period:
            return np.full(len(arr), np.nan)
        res = np.full(len(arr), np.nan)
        res[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            res[i] = (res[i-1] * (period-1) + arr[i]) / period
        return res
    
    atr_1d = rma(tr_1d, 14)
    plus_di_1d = 100 * rma(plus_dm_1d, 14) / atr_1d
    minus_di_1d = 100 * rma(minus_dm_1d, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = rma(dx_1d, 14)
    
    # Choppiness Index
    def choppiness_index(high, low, close, period=14):
        atr_sum = np.zeros(len(close))
        for i in range(len(close)):
            if i < period:
                atr_sum[i] = np.nan
                continue
            tr_sum = 0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], 
                         abs(high[j] - close[j-1]), 
                         abs(low[j] - close[j-1]))
                tr_sum += tr
            atr_sum[i] = tr_sum
        
        max_high = np.full(len(close), np.nan)
        min_low = np.full(len(close), np.nan)
        for i in range(len(close)):
            if i < period-1:
                max_high[i] = np.nan
                min_low[i] = np.nan
                continue
            max_high[i] = np.max(high[i-period+1:i+1])
            min_low[i] = np.min(low[i-period+1:i+1])
        
        chop = np.full(len(close), np.nan)
        for i in range(len(close)):
            if i < period-1 or atr_sum[i] == 0 or max_high[i] == min_low[i]:
                chop[i] = np.nan
                continue
            chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(period)
        return chop
    
    chop_1d = choppiness_index(high_1d, low_1d, close_1d, 14)
    
    # Regime: 1 = trending (ADX>25 AND Chop<61.8), -1 = ranging (Chop>=61.8), 0 = weak trend
    regime_1d = np.zeros(len(close_1d))
    regime_1d[(adx_1d > 25) & (chop_1d < 61.8)] = 1   # Trending
    regime_1d[chop_1d >= 61.8] = -1                    # Ranging
    
    regime_1d_aligned = align_htf_to_ltf(prices, df_1d, regime_1d)
    
    # === 6h Indicators: Elder Ray (Bull Power/Bear Power) ===
    # EMA(13) as proxy for equilibrium
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    bull_power = high - ema13
    bear_power = low - ema13
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(regime_1d_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5*ATR below entry (wider for 6h)
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.5*ATR above entry
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Regime filter: only trade in clear regimes
        if regime_1d_aligned[i] == 0:  # Weak trend - avoid
            signals[i] = 0.0
            continue
        
        # Volume confirmation: require volume spike
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            if regime_1d_aligned[i] == 1:  # Trending regime - trend follow
                # Strong bull power + price above EMA = long
                if bull_power[i] > 0 and price > ema13[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                # Strong bear power + price below EMA = short
                elif bear_power[i] < 0 and price < ema13[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            else:  # Ranging regime (regime = -1) - mean revert
                # Extreme bear power (oversold) + price below EMA = long
                if bear_power[i] < -np.std(bear_power[max(0, i-50):i]) * 1.5 and price < ema13[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                # Extreme bull power (overbought) + price above EMA = short
                elif bull_power[i] > np.std(bull_power[max(0, i-50):i]) * 1.5 and price > ema13[i]:
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