#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation
# - Bull Power = High - EMA13(close); Bear Power = EMA13(close) - Low
# - Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 (trending) AND volume > 1.5x 20-bar avg
# - Short when Bear Power > 0 AND Bull Power < 0 AND 1d ADX > 25 (trending) AND volume > 1.5x 20-bar avg
# - Exit when either power crosses zero (momentum shift) OR ADX < 20 (trend weakens)
# - Uses 1d ADX for regime filter to avoid choppy markets where Elder Ray fails
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Elder Ray captures momentum strength; ADX regime filter avoids false signals in ranging markets
# - Works in both bull (strong uptrends) and bear (strong downtrends) markets

name = "6h_1d_elder_ray_adx_regime_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Elder Ray components (13-period EMA)
    close_s = pd.Series(prices['close'])
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = prices['high'].values - ema13
    bear_power = ema13 - prices['low'].values
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute aligned 1d data
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Pre-compute 1d ADX(14) for regime filter
    # True Range
    tr1 = pd.Series(h_1d).diff().abs()
    tr2 = (pd.Series(h_1d) - pd.Series(c_1d).shift()).abs()
    tr3 = (pd.Series(l_1d) - pd.Series(c_1d).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = pd.Series(h_1d).diff()
    dm_minus = -pd.Series(l_1d).diff()
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0.0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0.0)
    
    # Smoothed DM and ATR
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_smooth = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / np.where(atr_smooth == 0, 1, atr_smooth)
    di_minus = 100 * dm_minus_smooth / np.where(atr_smooth == 0, 1, atr_smooth)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, 1, (di_plus + di_minus))
    adx14 = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to LTF
    adx14_aligned = align_htf_to_ltf(prices, df_1d, adx14)
    
    for i in range(20, n):  # Start after warmup periods
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_20_avg[i]) or np.isnan(adx14_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 (strong trend) AND volume spike
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                adx14_aligned[i] > 25 and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when Bear Power > 0 AND Bull Power < 0 AND 1d ADX > 25 (strong trend) AND volume spike
            elif (bear_power[i] > 0 and 
                  bull_power[i] < 0 and 
                  adx14_aligned[i] > 25 and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when power crosses zero (momentum shift) OR ADX < 20 (trend weakens)
            exit_signal = False
            if position == 1:  # Long position
                if bull_power[i] <= 0 or adx14_aligned[i] < 20:
                    exit_signal = True
            elif position == -1:  # Short position
                if bear_power[i] <= 0 or adx14_aligned[i] < 20:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals