#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and chop regime filter
# - Long when price breaks above 4h Donchian(20) high AND 12h volume > 1.5x 20-period volume SMA AND chop(14) < 38.2 (trending regime)
# - Short when price breaks below 4h Donchian(20) low AND 12h volume > 1.5x 20-period volume SMA AND chop(14) < 38.2 (trending regime)
# - Exit: price retreats to opposite Donchian band or chop > 61.8 (range regime)
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 20-50 trades/year on 4h timeframe to stay within fee drag limits
# - Uses Donchian channels for structure, chop regime filter to avoid whipsaws in ranging markets

name = "4h_12h_donchian_chop_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 4h Donchian channels (20-period)
    high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h volume SMA for confirmation
    volume_12h = df_12h['volume'].values
    volume_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    
    # Calculate 4h Chopiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR(1),14) / (max(high,14) - min(low,14))) / log10(14)
    tr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]  # First period TR
    atr1 = pd.Series(tr1).rolling(window=1, min_periods=1).sum().values  # ATR(1) is just TR
    sum_atr1_14 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * (np.log10(sum_atr1_14) - np.log10(max_high_14 - min_low_14)) / np.log10(14)
    
    for i in range(40, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(high_rolling_max[i]) or np.isnan(low_rolling_min[i]) or
            np.isnan(volume_sma_20_12h_aligned[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 1.5x 20-period volume SMA
        vol_confirm = volume_12h[i // 48] > 1.5 * volume_sma_20_12h_aligned[i] if i // 48 < len(volume_12h) else False
        
        # Regime filter: chop < 38.2 = trending regime (favor breakouts)
        trending_regime = chop[i] < 38.2
        ranging_regime = chop[i] > 61.8
        
        # Donchian breakout signals
        breakout_up = close[i] > high_rolling_max[i-1]  # Break above previous period high
        breakout_down = close[i] < low_rolling_min[i-1]  # Break below previous period low
        
        # Exit conditions: price retreats to opposite band or regime change to ranging
        exit_long = close[i] < low_rolling_min[i] or ranging_regime
        exit_short = close[i] > high_rolling_max[i] or ranging_regime
        
        if position == 0:  # Flat - look for entry
            if breakout_up and vol_confirm and trending_regime:
                position = 1
                signals[i] = 0.25
            elif breakout_down and vol_confirm and trending_regime:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals