#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Long when Bull Power > 0 AND Bear Power < 0 AND close > 1d EMA34 AND volume > 1.5x average
# Short when Bear Power < 0 AND Bull Power > 0 AND close < 1d EMA34 AND volume > 1.5x average
# Uses discrete sizing (0.25) and tight entry conditions to target 12-37 trades/year.
# Elder Ray measures bull/bear strength relative to trend; 1d EMA34 filters higher timeframe trend; volume confirms conviction.
# Timeframe: 6h (primary), HTF: 1d for EMA34 trend.

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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Elder Ray components on 6h timeframe
    # Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA34 and Elder Ray (need 13 for EMA13 + 34 for EMA34)
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_34_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Bull Power turns negative (weakening bullish momentum)
            # 2. Bear Power turns positive (emerging bearish pressure)
            # 3. Price crosses below 1d EMA34 (trend change)
            if (curr_bull_power <= 0 or
                curr_bear_power >= 0 or
                curr_close < curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Bear Power turns positive (weakening bearish momentum)
            # 2. Bull Power turns negative (emerging bullish pressure)
            # 3. Price crosses above 1d EMA34 (trend change)
            if (curr_bear_power >= 0 or
                curr_bull_power <= 0 or
                curr_close > curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Bull Power > 0 AND Bear Power < 0 AND close > 1d EMA34 AND volume confirm
            if (curr_bull_power > 0 and
                curr_bear_power < 0 and
                curr_close > curr_ema_34_1d and
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power < 0 AND Bull Power > 0 AND close < 1d EMA34 AND volume confirm
            elif (curr_bear_power < 0 and
                  curr_bull_power > 0 and
                  curr_close < curr_ema_34_1d and
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals