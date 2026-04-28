#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation.
# Bull Power = High - EMA(close), Bear Power = EMA(close) - Low.
# Enter long when Bull Power > 0 and rising, price > 1d EMA34 (uptrend), and volume > 1.5x 20-bar average.
# Enter short when Bear Power > 0 and rising, price < 1d EMA34 (downtrend), and volume > 1.5x 20-bar average.
# Exit when power becomes negative or price crosses EMA34.
# Uses discrete position sizing (0.25) to manage drawdown in volatile markets.
# Target: 80-120 total trades over 4 years (20-30/year) to minimize fee drag.
# Elder Ray measures buying/selling pressure relative to trend; combining with 1d EMA34 ensures alignment with higher timeframe trend;
# volume confirmation filters weak breakouts. Works in bull (strong buying pressure) and bear (strong selling pressure).

name = "6h_ElderRay_BullBearPower_1dEMA34_Trend_VolumeConfirm_v1"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 6h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate EMA of close for 6h (needed for Elder Ray)
    # Use sufficient history for EMA calculation
    close_series = pd.Series(close)
    ema_close = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Elder Ray components
    bull_power = high - ema_close  # Buying pressure
    bear_power = ema_close - low   # Selling pressure
    
    # Volume confirmation: >1.5x 20-bar average (moderate to balance frequency)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Elder Ray momentum: rising power (current > previous)
        bull_power_rising = bull_power[i] > bull_power[i-1]
        bear_power_rising = bear_power[i] > bear_power[i-1]
        
        # Price relative to 1d EMA34
        price_above_ema = close[i] > ema_34_aligned[i]
        price_below_ema = close[i] < ema_34_aligned[i]
        
        # Entry conditions
        enter_long = bull_power[i] > 0 and bull_power_rising and price_above_ema and vol_confirm
        enter_short = bear_power[i] > 0 and bear_power_rising and price_below_ema and vol_confirm
        
        # Exit conditions: power turns negative or price crosses EMA34
        exit_long = bull_power[i] <= 0 or not price_above_ema
        exit_short = bear_power[i] <= 0 or not price_below_ema
        
        # Handle entries and exits
        if enter_long and position <= 0:
            signals[i] = 0.25
            position = 1
        elif enter_short and position >= 0:
            signals[i] = -0.25
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals