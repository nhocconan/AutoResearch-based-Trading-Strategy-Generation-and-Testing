#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + Elder Ray with 1w EMA50 trend filter
# Uses discrete sizing 0.25 to balance profit and fee drag. Target: 30-100 total trades over 4 years (7-25/year).
# Williams Alligator (jaw/teeth/lips) identifies trend absence/presence. Elder Ray (bull/bear power) measures trend strength.
# 1w EMA50 filters counter-trend moves on weekly timeframe. Strategy designed to avoid whipsaws in ranging markets
# while capturing strong trends in both bull and bear markets via weekly trend alignment.

name = "1d_WilliamsAlligator_ElderRay_1wEMA50_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate Williams Alligator (SMAs with smoothing)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw_shift = 8
    teeth_shift = 5
    lips_shift = 3
    
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=jaw_period, min_periods=jaw_period).mean().shift(jaw_shift).values
    teeth = pd.Series(median_price).rolling(window=teeth_period, min_periods=teeth_period).mean().shift(teeth_shift).values
    lips = pd.Series(median_price).rolling(window=lips_period, min_periods=lips_period).mean().shift(lips_shift).values
    
    # Calculate Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate 1w EMA(50) for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average (moderate to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 20, 13, 50, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready or outside session
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(atr_14[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_atr = atr_14[i]
        
        # Alligator sleeping condition: all lines intertwined (market ranging)
        alligator_sleeping = (abs(curr_jaw - curr_teeth) < (curr_atr * 0.5) and 
                             abs(curr_teeth - curr_lips) < (curr_atr * 0.5) and
                             abs(curr_lips - curr_jaw) < (curr_atr * 0.5))
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with Elder Ray confirmation and weekly trend filter
            if curr_volume_spike and not alligator_sleeping:
                # Bullish: Bull Power positive + price above Alligator lips + price above weekly EMA50
                if curr_bull_power > 0 and curr_close > curr_lips and curr_close > curr_ema_50_1w:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish: Bear Power negative + price below Alligator teeth + price below weekly EMA50
                elif curr_bear_power < 0 and curr_close < curr_teeth and curr_close < curr_ema_50_1w:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2 * ATR below entry
            stop_loss = entry_price - 2.0 * curr_atr
            # Exit: Stoploss hit OR Elder Ray turns negative OR price crosses below Alligator teeth
            if curr_low <= stop_loss or curr_bear_power < 0 or curr_close < curr_teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_loss = entry_price + 2.0 * curr_atr
            # Exit: Stoploss hit OR Elder Ray turns positive OR price crosses above Alligator lips
            if curr_high >= stop_loss or curr_bull_power > 0 or curr_close > curr_lips:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals