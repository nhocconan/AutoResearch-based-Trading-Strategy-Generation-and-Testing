#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume spike + choppiness regime filter
# - Long: price breaks above 20-period 12h Donchian high with volume > 2x 20-period average and CHOP > 61.8 (range)
# - Short: price breaks below 20-period 12h Donchian low with volume > 2x 20-period average and CHOP > 61.8
# - Exit: price returns to opposite Donchian level (mean reversion at channel)
# - Uses 1d ATR for CHOP calculation and volume spike confirmation
# - Works in both bull and bear markets by fading extremes in ranging regimes
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits

name = "12h_1d_donchian_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for ATR and volume (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Pre-compute 1d ATR for choppiness calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 1d volume for spike confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d ATR and volume SMA to 12h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute 12h Donchian channels (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 12h volume for spike confirmation
    volume = prices['volume'].values
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(volume_sma_20_1d_aligned[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Choppiness Index calculation (using 1d ATR)
        # CHOP = 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
        # Simplified: we use ATR ratio to detect ranging markets
        # For 12h timeframe, we approximate using 1d ATR over 14 periods
        atr_sum = atr_1d_aligned[i] * 14  # Approximation for chop calculation
        period_high = high_s.rolling(window=14, min_periods=14).max().iloc[i] if hasattr(high_s.rolling(window=14, min_periods=14).max(), 'iloc') else high_s[-14+i: i+1].max() if i >= 13 else high_s[:i+1].max()
        period_low = low_s.rolling(window=14, min_periods=14).min().iloc[i] if hasattr(low_s.rolling(window=14, min_periods=14).min(), 'iloc') else low_s[-14+i: i+1].min() if i >= 13 else low_s[:i+1].min()
        if i >= 13:
            period_high_val = high_s[i-13:i+1].max()
            period_low_val = low_s[i-13:i+1].min()
        else:
            period_high_val = high_s[:i+1].max()
            period_low_val = low_s[:i+1].min()
        
        period_range = period_high_val - period_low_val
        chop = 0.0
        if period_range > 0 and atr_sum > 0:
            chop = 100 * np.log10(atr_sum / period_range) / np.log10(14)
        
        # Volume confirmation: current volume > 2x 20-period average (both 12h and 1d)
        vol_confirm_12h = volume_current > 2.0 * volume_sma_20[i]
        vol_confirm_1d = volume_1d[i] > 2.0 * volume_sma_20_1d_aligned[i] if i < len(volume_1d) else False
        vol_confirm = vol_confirm_12h and vol_confirm_1d
        
        # Regime filter: CHOP > 61.8 indicates ranging market (mean reversion favorable)
        ranging_regime = chop > 61.8
        
        # Donchian levels
        upper_channel = donchian_high[i]
        lower_channel = donchian_low[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price breaks above upper channel with volume confirmation in ranging market
        if close_price > upper_channel and vol_confirm and ranging_regime:
            enter_long = True
        
        # Short breakout: price breaks below lower channel with volume confirmation in ranging market
        if close_price < lower_channel and vol_confirm and ranging_regime:
            enter_short = True
        
        # Exit conditions: mean reversion at opposite Donchian level
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price drops back to lower channel
            exit_long = close_price <= lower_channel
        elif position == -1:
            # Exit short if price rises back to upper channel
            exit_short = close_price >= upper_channel
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals