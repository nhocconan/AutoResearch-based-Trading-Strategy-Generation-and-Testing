#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Index with 12-hour EMA trend filter and 1-day volume confirmation
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# Long when Bull Power > 0 AND Bear Power < 0 AND price > EMA(20) AND volume > 1.5x 20-period average
# Short when Bull Power < 0 AND Bear Power > 0 AND price < EMA(20) AND volume > 1.5x 20-period average
# Exit when Bull Power and Bear Power have same sign (both positive or both negative)
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 12-hour EMA for trend filter and 1-day volume for confirmation
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_elder_ray_12h_ema_1d_vol_v4"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12-hour data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 1-day data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 12-hour EMA(20) for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 1-day volume average (20-period)
    volume_1d = df_1d['volume'].values
    volume_1d_s = pd.Series(volume_1d)
    volume_ma = volume_1d_s.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    
    # Elder Ray components: EMA(13) of close
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Bull Power and Bear Power both positive (bullish exhaustion) 
            # or both negative (bearish takeover)
            elif (bull_power[i] > 0 and bear_power[i] > 0) or (bull_power[i] < 0 and bear_power[i] < 0):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Bull Power and Bear Power both positive (bullish takeover) 
            # or both negative (bearish exhaustion)
            elif (bull_power[i] > 0 and bear_power[i] > 0) or (bull_power[i] < 0 and bear_power[i] < 0):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Elder Ray divergence with trend and volume filter
            # Trend filter: price above/below 12-hour EMA(20)
            uptrend = close[i] > ema_12h_aligned[i]
            downtrend = close[i] < ema_12h_aligned[i]
            # Volume filter: volume > 1.5x 20-period average
            volume_filter = volume[i] > 1.5 * volume_ma_aligned[i]
            
            # Long: Bull Power > 0 (buying strength) AND Bear Power < 0 (weak selling) 
            # AND uptrend AND volume confirmation
            if bull_power[i] > 0 and bear_power[i] < 0 and uptrend and volume_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Bull Power < 0 (weak buying) AND Bear Power > 0 (selling strength) 
            # AND downtrend AND volume confirmation
            elif bull_power[i] < 0 and bear_power[i] > 0 and downtrend and volume_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals