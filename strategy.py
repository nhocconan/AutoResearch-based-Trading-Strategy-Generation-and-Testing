#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d EMA34 trend + volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: bull power = high - EMA13, bear power = low - EMA13
# Long when bull power > 0 and increasing + price > 1d EMA34 + volume spike
# Short when bear power < 0 and decreasing + price < 1d EMA34 + volume spike
# Works in bull/bear via 1d EMA34 trend filter. Target: 12-37 trades/year (50-150 total over 4 years).

name = "6h_ElderRay_1dEMA34_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate EMA13 for Elder Ray (6h timeframe)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull power: high - EMA13
    bear_power = low - ema_13   # Bear power: low - EMA13
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 34, 20, 13)  # warmup for EMA34, EMA13, volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_34_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Calculate Elder Ray momentum (change from previous bar)
        bull_power_mom = curr_bull_power - bull_power[i-1] if i > 0 else 0
        bear_power_mom = curr_bear_power - bear_power[i-1] if i > 0 else 0
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Bull power turns negative (bearish pressure)
            # 2. Price crosses below 1d EMA34 (trend change)
            # 3. Volume confirmation fails (weak breakout)
            if (curr_bull_power <= 0 or
                curr_close < curr_ema_34_1d or
                not curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Bear power turns positive (bullish pressure)
            # 2. Price crosses above 1d EMA34 (trend change)
            # 3. Volume confirmation fails (weak breakout)
            if (curr_bear_power >= 0 or
                curr_close > curr_ema_34_1d or
                not curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: bull power > 0 AND increasing + price > 1d EMA34 + volume confirm
            if (curr_bull_power > 0 and
                bull_power_mom > 0 and
                curr_close > curr_ema_34_1d and
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: bear power < 0 AND decreasing + price < 1d EMA34 + volume confirm
            elif (curr_bear_power < 0 and
                  bear_power_mom < 0 and
                  curr_close < curr_ema_34_1d and
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals