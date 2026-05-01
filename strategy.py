#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + 1w EMA trend filter + volume confirmation.
# Long when: price > Alligator Jaw (13-period SMA shifted 8) AND Alligator Teeth > Alligator Lips AND price > 1w EMA34 AND 1d volume > 1.5x 20-period average
# Short when: price < Alligator Jaw AND Alligator Teeth < Alligator Lips AND price < 1w EMA34 AND 1d volume > 1.5x 20-period average
# Uses discrete sizing 0.25. Target: 15-30 trades/year.
# Williams Alligator identifies trend absence/presence via smoothed SMAs. 1w EMA ensures higher-timeframe trend alignment.
# Volume confirms conviction. Works in bull (trend continuation) and bear (trend continuation) by trading with the 1w trend.

name = "1d_WilliamsAlligator_1wEMA34_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for Alligator and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for EMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Williams Alligator on 1d: Jaw (13,8), Teeth (8,5), Lips (5,3)
    # Smoothed SMA with future shift
    close_1d = df_1d['close'].values
    
    # Jaw: 13-period SMA shifted 8 bars
    sma_13 = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    jaw = np.concatenate([np.full(8, np.nan), sma_13[:-8]]) if len(sma_13) > 8 else np.full_like(sma_13, np.nan)
    
    # Teeth: 8-period SMA shifted 5 bars
    sma_8 = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    teeth = np.concatenate([np.full(5, np.nan), sma_8[:-5]]) if len(sma_8) > 5 else np.full_like(sma_8, np.nan)
    
    # Lips: 5-period SMA shifted 3 bars
    sma_5 = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    lips = np.concatenate([np.full(3, np.nan), sma_5[:-3]]) if len(sma_5) > 3 else np.full_like(sma_5, np.nan)
    
    # 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 1d volume confirmation: volume > 1.5x 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to LTF (1d)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for longest indicator
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_vol = volume[i]
        curr_jaw = jaw_aligned[i]
        curr_teeth = teeth_aligned[i]
        curr_lips = lips_aligned[i]
        curr_ema_1w = ema_34_1w_aligned[i]
        curr_vol_ma = vol_ma_20_aligned[i]
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = curr_vol > (curr_vol_ma * 1.5)
        
        # Alligator conditions: Teeth > Lips for bullish alignment, Teeth < Lips for bearish
        bullish_alligator = curr_teeth > curr_lips
        bearish_alligator = curr_teeth < curr_lips
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price > Jaw AND bullish Alligator AND price > 1w EMA AND volume confirmed
            if (curr_close > curr_jaw and 
                bullish_alligator and 
                curr_close > curr_ema_1w and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price < Jaw AND bearish Alligator AND price < 1w EMA AND volume confirmed
            elif (curr_close < curr_jaw and 
                  bearish_alligator and 
                  curr_close < curr_ema_1w and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < Jaw OR Alligator turns bearish (Teeth < Lips) OR price < 1w EMA
            if (curr_close < curr_jaw or 
                not bullish_alligator or  # Teeth <= Lips
                curr_close < curr_ema_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > Jaw OR Alligator turns bullish (Teeth > Lips) OR price > 1w EMA
            if (curr_close > curr_jaw or 
                not bearish_alligator or  # Teeth >= Lips
                curr_close > curr_ema_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals