#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Elder Ray (Bull/Bear Power) with 1d EMA(34) trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13. Strong bull power + price above 1d EMA(34) + volume spike = long
# Strong bear power + price below 1d EMA(34) + volume spike = short. Works in both bull/bear markets by measuring
# underlying strength/weakness. Target: 12-37 trades/year on 6h.

name = "6h_12hElderRay_BullBearPower_1dEMA34_VolumeSpike_v1"
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
    
    # Load 12h data ONCE before loop for Elder Ray calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA(13) for Elder Ray
    close_12h_s = pd.Series(df_12h['close'].values)
    ema13_12h = close_12h_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 12h Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_12h = df_12h['high'].values - ema13_12h
    bear_power_12h = df_12h['low'].values - ema13_12h
    
    # Align 12h Elder Ray components to 6h timeframe (wait for 12h bar to close)
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power_12h)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power_12h)
    
    # Calculate 1d EMA(34) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    close_1d_s = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA(34)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 1.5x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i]) if i >= 20 else np.mean(volume[:i]) if i > 0 else 0
        volume_spike = volume[i] > (1.5 * vol_ma_20) if i > 0 else False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        curr_bull = bull_power_aligned[i]
        curr_bear = bear_power_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: strong bull power + price above 1d EMA(34)
                if curr_bull > 0 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: strong bear power + price below 1d EMA(34)
                elif curr_bear < 0 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR bull power turns negative (reversal signal)
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_bull <= 0:
                signals[i] = 0.0
                position = 0
            # Take profit: reduce position when bear power becomes significantly negative
            elif curr_bear < -curr_atr * 0.5:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR bear power turns positive (reversal signal)
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_bear >= 0:
                signals[i] = 0.0
                position = 0
            # Take profit: reduce position when bull power becomes significantly positive
            elif curr_bull > curr_atr * 0.5:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals