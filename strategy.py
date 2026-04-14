#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily 144-period EMA as trend filter and 20-period ATR for volatility-based position sizing.
# Long when price crosses above EMA144 with expanding volatility (ATR > ATR_ma) and volume confirmation.
# Short when price crosses below EMA144 with same conditions.
# Exit when price crosses back below/above EMA144.
# Uses daily EMA for trend (avoids whipsaw), ATR for volatility filter (avoids low-vol chop), volume for confirmation.
# Target: 20-30 trades/year per symbol (80-120 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 144:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA(144)
    ema_144 = pd.Series(close_1d).ewm(span=144, adjust=False, min_periods=144).mean().values
    
    # Load daily data for ATR calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(20) on daily
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_20 = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Load daily data for ATR moving average
    atr_ma = pd.Series(atr_20).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align indicators to lower timeframe
    ema_144_aligned = align_htf_to_ltf(prices, df_1d, ema_144)
    atr_20_aligned = align_htf_to_ltf(prices, df_1d, atr_20)
    atr_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_ma)
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(144, 20, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_144_aligned[i]) or 
            np.isnan(atr_20_aligned[i]) or
            np.isnan(atr_ma_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: expanding ATR
        volatility_expanding = atr_20_aligned[i] > atr_ma_aligned[i]
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # Trend filter: price relative to EMA144
        price_above_ema = close[i] > ema_144_aligned[i]
        price_below_ema = close[i] < ema_144_aligned[i]
        
        if position == 0:
            # Look for EMA crossovers with filters
            # Long: price crosses above EMA144 with volatility expansion and volume
            if (price_above_ema and 
                not (close[i-1] > ema_144_aligned[i-1]) and  # Was below or equal
                volatility_expanding and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price crosses below EMA144 with volatility expansion and volume
            elif (price_below_ema and 
                  not (close[i-1] < ema_144_aligned[i-1]) and  # Was above or equal
                  volatility_expanding and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back below EMA144
            if close[i] < ema_144_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses back above EMA144
            if close[i] > ema_144_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_DailyEMA144_ATR_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0