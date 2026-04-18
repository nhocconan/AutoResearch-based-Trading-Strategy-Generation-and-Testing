#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for calculations (HTF)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h ATR (14-period) for volatility filter
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h EMA34 for trend filter
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 12h volume spike (volume > 2.0x 20-period average)
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike_12h = volume_12h > (2.0 * vol_ma_12h)
    
    # Align indicators to 4h timeframe
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    volume_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h.astype(float))
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_12h_aligned[i]) or
            np.isnan(atr_12h_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR is above its 30-period average
        if i >= 30:
            atr_ma_12h = pd.Series(atr_12h).rolling(window=30, min_periods=30).mean().values
            atr_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_ma_12h)
            vol_filter = atr_12h_aligned[i] > atr_ma_12h_aligned[i] if not np.isnan(atr_ma_12h_aligned[i]) else False
        else:
            vol_filter = False
        
        trade_allowed = volume_spike_12h_aligned[i] and vol_filter
        
        if position == 0:
            # Long: Donchian breakout above upper band with EMA34 uptrend
            if trade_allowed and close[i] > donchian_high[i] and close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown below lower band with EMA34 downtrend
            elif trade_allowed and close[i] < donchian_low[i] and close[i] < ema34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below EMA34 or Donchian lower band
            if close[i] < ema34_12h_aligned[i] or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above EMA34 or Donchian upper band
            if close[i] > ema34_12h_aligned[i] or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA34_VolumeSpike_ATRFilter_v1"
timeframe = "4h"
leverage = 1.0