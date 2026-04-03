#!/usr/bin/env python3
"""
Experiment #217: 4h Donchian Breakout + HMA Trend + Volume Confirmation + ATR Stoploss
HYPOTHESIS: Donchian channel breakouts on 4h timeframe, filtered by higher timeframe (1d/1w) HMA trend and volume spikes, capture institutional breakout moves in both bull and bear markets. The strategy uses tight entry conditions (breakout + trend + volume) to limit trades to 75-200 total over 4 years, minimizing fee drag while maintaining edge. ATR-based stoploss manages risk. Works in bull markets via breakout continuation and in bear markets via breakdown continuation when aligned with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_217_4h_donchian_hma_volume_1d_1w_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # HMA(21) on 1d close: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def hma(arr, period):
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma2 = pd.Series(arr).ewm(span=half, adjust=False).mean()
        wma1 = pd.Series(arr).ewm(span=period, adjust=False).mean()
        raw = 2 * wma2 - wma1
        return pd.Series(raw).ewm(span=sqrt, adjust=False).mean().values
    hma_21_1d = hma(close_1d, 21)
    trend_up_1d = close_1d > hma_21_1d
    trend_down_1d = close_1d < hma_21_1d
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    trend_down_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_down_1d)
    
    # === HTF: 1w data for regime filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    hma_21_1w = hma(close_1w, 21)
    regime_bull = close_1w > hma_21_1w  # Bull regime when price above weekly HMA
    regime_bear = close_1w < hma_21_1w  # Bear regime when price below weekly HMA
    regime_bull_aligned = align_htf_to_ltf(prices, df_1w, regime_bull)
    regime_bear_aligned = align_htf_to_ltf(prices, df_1w, regime_bear)
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr_4h = np.zeros(n)
    tr_4h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_4h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr_4h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === 4h Indicators: Donchian Channel (20) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # Enough for Donchian(20), volume MA(20), ATR(14)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(trend_up_1d_aligned[i]) or np.isnan(trend_down_1d_aligned[i]) or
            np.isnan(regime_bull_aligned[i]) or np.isnan(regime_bear_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]  # Wider stop for 4h volatility
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit at 2.5x ATR profit
                if price >= entry_price + 2.5 * atr_14[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit at 2.5x ATR profit
                if price <= entry_price - 2.5 * atr_14[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout above upper band + 1d uptrend + volume spike
        # In bear regime, only take long if also aligned with weekly bull regime (counter-trend bounce)
        if (price > donchian_high[i] and 
            trend_up_1d_aligned[i] and 
            volume_spike and
            (regime_bull_aligned[i] or (regime_bear_aligned[i] and regime_bull_aligned[i]))):  # Allow counter-trend in bear
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Donchian breakdown below lower band + 1d downtrend + volume spike
        # In bull regime, only take short if also aligned with weekly bear regime (counter-trend sell)
        elif (price < donchian_low[i] and 
              trend_down_1d_aligned[i] and 
              volume_spike and
              (regime_bear_aligned[i] or (regime_bull_aligned[i] and regime_bear_aligned[i]))):  # Allow counter-trend in bull
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals