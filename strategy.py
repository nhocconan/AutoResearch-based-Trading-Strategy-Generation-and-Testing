#!/usr/bin/env python3
"""
Experiment #2866: 4h Donchian(20) breakout + HMA(21) trend + Volume confirmation + ATR stoploss
HYPOTHESIS: 4h Donchian breakouts capture strong momentum moves. HMA(21) filter ensures we 
only trade in the direction of the intermediate trend, reducing whipsaw. Volume confirmation 
(>1.5x average) validates breakout strength. ATR-based stoploss manages risk. 4h timeframe 
targets 20-50 trades/year, balancing opportunity with fee drag. Works in both bull (breakouts) 
and bear (breakdowns) markets by trading both directions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2866_4h_donchian20_hma21_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)  # 1 = uptrend, -1 = downtrend
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 4h Indicators: Donchian channels (20) ===
    donchian_window = 20
    highest_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # === 4h Indicators: HMA(21) for trend confirmation ===
    def hull_moving_average(arr, period):
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        if half_period < 1:
            half_period = 1
        if sqrt_period < 1:
            sqrt_period = 1
        wma1 = pd.Series(arr).rolling(window=half_period, min_periods=half_period).mean().values
        wma2 = pd.Series(arr).rolling(window=period, min_periods=period).mean().values
        raw_hma = 2 * wma1 - wma2
        hma = pd.Series(raw_hma).rolling(window=sqrt_period, min_periods=sqrt_period).mean().values
        return hma
    
    hma_21 = hull_moving_average(close, 21)
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === ATR(14) for stoploss ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    stop_loss_price = 0.0
    
    warmup = max(donchian_window, 21, 20, 14, 50)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(hma_21[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(trend_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Check stoploss hit
            if position_side > 0 and price <= stop_loss_price:
                in_position = False
                position_side = 0
                signals[i] = 0.0
                continue
            elif position_side < 0 and price >= stop_loss_price:
                in_position = False
                position_side = 0
                signals[i] = 0.0
                continue
            
            # Check Donchian opposite touch (take profit/exit)
            if position_side > 0 and price <= lowest_low[i]:
                in_position = False
                position_side = 0
                signals[i] = 0.0
                continue
            elif position_side < 0 and price >= highest_high[i]:
                in_position = False
                position_side = 0
                signals[i] = 0.0
                continue
            
            # Hold position
            signals[i] = SIZE * position_side
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation
        volume_confirm = vol_ratio[i] > 1.5
        
        if volume_confirm:
            # Get trend filters
            hma_trend = 1 if close[i] > hma_21[i] else -1
            daily_trend = trend_1d_aligned[i]
            
            # Long entry: price breaks above Donchian upper + HMA up + daily trend up
            if (price > highest_high[i] and 
                hma_trend > 0 and 
                daily_trend > 0):
                in_position = True
                position_side = 1
                entry_price = close[i]
                stop_loss_price = entry_price - 2.0 * atr[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower + HMA down + daily trend down
            elif (price < lowest_low[i] and 
                  hma_trend < 0 and 
                  daily_trend < 0):
                in_position = True
                position_side = -1
                entry_price = close[i]
                stop_loss_price = entry_price + 2.0 * atr[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals