#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily ATR (14-period) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate daily volume spike (volume > 2.0x 20-period average)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    
    # Align indicators to 1d timeframe (daily resolution)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Calculate daily Donchian channels (20-period)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 1d timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_1d_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR is above its 30-period average
        if i >= 30:
            atr_ma_1d = pd.Series(atr_1d).rolling(window=30, min_periods=30).mean().values
            atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
            vol_filter = atr_1d_aligned[i] > atr_ma_1d_aligned[i] if not np.isnan(atr_ma_1d_aligned[i]) else False
        else:
            vol_filter = False
        
        trade_allowed = volume_spike_1d_aligned[i] and vol_filter
        
        if position == 0:
            # Long: Donchian breakout above upper band with EMA34 uptrend
            if trade_allowed and close[i] > donchian_high_aligned[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown below lower band with EMA34 downtrend
            elif trade_allowed and close[i] < donchian_low_aligned[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below EMA34 or Donchian lower band
            if close[i] < ema34_1d_aligned[i] or close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above EMA34 or Donchian upper band
            if close[i] > ema34_1d_aligned[i] or close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1dEMA34_VolumeSpike_ATRFilter"
timeframe = "1d"
leverage = 1.0