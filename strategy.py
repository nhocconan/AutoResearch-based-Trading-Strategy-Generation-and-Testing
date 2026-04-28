#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA34 trend filter and volume confirmation
# Williams Alligator: Jaw=SMA(13,8), Teeth=SMA(8,5), Lips=SMA(5,3)
# Long when Lips > Teeth > Jaw (bullish alignment) and price > 1w EMA34 and volume > 2.0x 20-bar average
# Short when Lips < Teeth < Jaw (bearish alignment) and price < 1w EMA34 and volume > 2.0x 20-bar average
# Uses 1d timeframe targeting 7-25 trades/year (~30-100 total over 4 years) to minimize fee drag.
# Alligator identifies trends; 1w EMA34 filter ensures higher-timeframe alignment; volume avoids chop.

name = "1d_WilliamsAlligator_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Williams Alligator components (1d)
    # Jaw: SMA(13,8) - 13-period SMA shifted 8 bars ahead
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8)
    # Teeth: SMA(8,5) - 8-period SMA shifted 5 bars ahead
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5)
    # Lips: SMA(5,3) - 5-period SMA shifted 3 bars ahead
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3)
    
    jaw_vals = jaw.values
    teeth_vals = teeth.values
    lips_vals = lips.values
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(34, 20, 13, 8, 5)  # EMA34, volume MA20, Alligator shifts
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(lips_vals[i]) or 
            np.isnan(teeth_vals[i]) or np.isnan(jaw_vals[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        lips_val = lips_vals[i]
        teeth_val = teeth_vals[i]
        jaw_val = jaw_vals[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Lips > Teeth > Jaw (bullish alignment), price above 1w EMA34, volume spike
            if lips_val > teeth_val and teeth_val > jaw_val and price > ema_34_1w_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: Lips < Teeth < Jaw (bearish alignment), price below 1w EMA34, volume spike
            elif lips_val < teeth_val and teeth_val < jaw_val and price < ema_34_1w_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stoploss or Alligator sleeping (crossing)
            # ATR-based stoploss: 2.0 * ATR below entry (using 1d ATR)
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price - 2.0 * atr_val
            # Exit on stoploss or when Alligator components cross (trend weakening)
            if price < stop_loss or lips_val <= teeth_val or teeth_val <= jaw_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on stoploss or Alligator sleeping (crossing)
            # ATR-based stoploss: 2.0 * ATR above entry
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price + 2.0 * atr_val
            # Exit on stoploss or when Alligator components cross (trend weakening)
            if price > stop_loss or lips_val >= teeth_val or teeth_val >= jaw_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals