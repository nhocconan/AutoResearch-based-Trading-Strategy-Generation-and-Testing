#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Elder Ray (Bull/Bear Power) with 1w trend filter and volume confirmation
# Elder Ray measures bull/bear power via EMA(13): Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# In bull markets (1w Uptrend): Buy when Bull Power turns positive with volume confirmation
# In bear markets (1w Downtrend): Sell when Bear Power turns negative with volume confirmation
# Uses 6h timeframe for lower frequency (target: 12-37 trades/year) to minimize fee drag
# 1w EMA(34) determines regime, 1d EMA(13) for Elder Ray calculation

name = "6h_ElderRay_BullBearPower_1wTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(13) for Elder Ray
    close_1d_s = pd.Series(df_1d['close'].values)
    ema_13_1d = close_1d_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema_13_1d  # Bull Power = High - EMA(13)
    bear_power = low_1d - ema_13_1d   # Bear Power = Low - EMA(13)
    
    # Align Elder Ray to 6h timeframe (wait for completed 1d bar)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 1w EMA(34) for trend filter
    close_1w_s = pd.Series(df_1w['close'].values)
    ema_34_1w = close_1w_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate ATR(14) for dynamic stoploss on 6h timeframe
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 34  # warmup for EMA(34)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 1.8x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (1.8 * vol_ma_20)
        
        curr_close = close[i]
        curr_atr = atr[i]
        curr_bull = bull_power_aligned[i]
        curr_bear = bear_power_aligned[i]
        curr_ema_34_1w = ema_34_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Bullish entry: 1w Uptrend AND Bull Power turns positive (bullish momentum)
                if curr_ema_34_1w > 0 and curr_bull > 0:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: 1w Downtrend AND Bear Power turns negative (bearish momentum)
                elif curr_ema_34_1w < 0 and curr_bear < 0:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR Bear Power turns negative (momentum loss)
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_bear < 0:  # Bear Power negative = bearish momentum
                signals[i] = 0.0
                position = 0
            # Take profit: Bull Power exceeds 2x ATR (strong momentum)
            elif curr_bull > (2.0 * curr_atr):
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR Bull Power turns positive (momentum loss)
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_bull > 0:  # Bull Power positive = bullish momentum
                signals[i] = 0.0
                position = 0
            # Take profit: Bear Power exceeds 2x ATR (strong momentum)
            elif abs(curr_bear) > (2.0 * curr_atr):
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals