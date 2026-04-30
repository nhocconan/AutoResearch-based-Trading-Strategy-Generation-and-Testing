#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull Power/Bear Power) with 1d EMA34 trend filter and volume confirmation.
# Elder Ray measures bull/bear strength relative to EMA13. Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Long when Bull Power > 0 and rising, price above 1d EMA34, volume > 1.5x 20-period average.
# Short when Bear Power < 0 and falling, price below 1d EMA34, volume > 1.5x 20-period average.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-30 trades/year.

name = "6h_ElderRay_BullBearPower_1dEMA34_VolumeConfirm_ATRStop_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d data
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate EMA13 for Elder Ray (6h timeframe)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Calculate ATR(14) for 6h timeframe stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 100  # warmup for EMA and ATR
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_ema = ema_34_aligned[i]
        curr_atr = atr[i]
        
        # Volume confirmation: volume > 1.5x 20-period average (balanced threshold)
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > (1.5 * vol_ma_20)
        else:
            volume_confirm = False
        
        # Elder Ray momentum: rising Bull Power or falling Bear Power
        if i >= 1:
            bull_power_rising = curr_bull_power > bull_power[i-1]
            bear_power_falling = curr_bear_power < bear_power[i-1]
        else:
            bull_power_rising = False
            bear_power_falling = False
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 and rising, price above 1d EMA34, volume spike
            if (curr_bull_power > 0 and 
                bull_power_rising and 
                curr_close > curr_ema and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Bear Power < 0 and falling, price below 1d EMA34, volume spike
            elif (curr_bear_power < 0 and 
                  bear_power_falling and 
                  curr_close < curr_ema and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit conditions: Bull Power <= 0 OR stoploss hit
            if (curr_bull_power <= 0 or 
                curr_close < entry_price - 2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: Bear Power >= 0 OR stoploss hit
            if (curr_bear_power >= 0 or 
                curr_close > entry_price + 2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals