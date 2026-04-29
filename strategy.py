#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA50 trend filter and volume spike confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Only take longs when Bull Power > 0 and increasing (momentum) in uptrend (price > 1d EMA50)
# Only take shorts when Bear Power < 0 and decreasing (momentum) in downtrend (price < 1d EMA50)
# Volume confirmation (>1.8x 20-period average) filters weak signals
# Designed for ~15-30 trades/year on 6h timeframe to minimize fee drag while capturing strong momentum moves
# Works in both bull and bear markets via 1d trend filter - only trades in trend direction

name = "6h_ElderRay_BullBearPower_1dEMA50_VolumeSpike_v1"
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
    
    # Get 1d data for EMA50 trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA13 for Elder Ray (on 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Calculate 20-period average volume for confirmation (on 6h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR (14-period) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # Volume MA and ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_ema13 = ema_13[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        curr_atr = atr[i]
        
        # Calculate Elder Ray momentum (change from previous bar)
        if i > start_idx:
            bull_power_momentum = curr_bull_power - bull_power[i-1]
            bear_power_momentum = curr_bear_power - bear_power[i-1]
        else:
            bull_power_momentum = 0.0
            bear_power_momentum = 0.0
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit or Elder Ray turns negative (loss of bullish momentum)
            if curr_close < entry_price - 2.5 * curr_atr or curr_bull_power <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or Elder Ray turns positive (loss of bearish momentum)
            if curr_close > entry_price + 2.5 * curr_atr or curr_bear_power >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.8x 20-period average
            vol_confirm = curr_volume > 1.8 * curr_vol_ma
            
            # Long entry: Bull Power > 0 and increasing (momentum) in uptrend (price > 1d EMA50)
            if vol_confirm and curr_close > curr_ema50_1d:
                if curr_bull_power > 0 and bull_power_momentum > 0:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
            # Short entry: Bear Power < 0 and decreasing (momentum) in downtrend (price < 1d EMA50)
            elif vol_confirm and curr_close < curr_ema50_1d:
                if curr_bear_power < 0 and bear_power_momentum < 0:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals