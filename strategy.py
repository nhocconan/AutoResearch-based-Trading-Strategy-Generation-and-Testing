#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and choppiness regime filter.
# Long when price breaks above Donchian(20) high AND volume > 2.0x 20-bar average AND chop > 61.8 (range) → mean reversion long at lower band.
# Short when price breaks below Donchian(20) low AND volume > 2.0x 20-bar average AND chop < 38.2 (trend) → trend continuation short.
# Uses discrete sizing 0.25 to manage drawdown. Target: 75-200 total trades over 4 years (19-50/year).
# Choppiness filter avoids false breakouts in strong trends and choppy markets.
# Volume spike confirms institutional participation.
# Primary timeframe: 4h, HTF: 1d for Donchian structure and volume/chop.

name = "4h_Donchian20_VolumeSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Donchian, volume MA, and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # need 20 for Donchian + 14 for chop
        return np.zeros(n)
    
    # Donchian(20) channels from 1d data (structure)
    donchian_high_raw = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low_raw = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume MA(20) for confirmation
    vol_ma_raw = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index(14) for regime filter
    # CHOP = 100 * log10(sum(TR(14)) / (log10(HH(14)-LL(14)) * 14))
    tr1 = np.maximum(df_1d['high'].values, np.roll(df_1d['close'].values, 1)) - np.minimum(df_1d['low'].values, np.roll(df_1d['close'].values, 1))
    tr1[0] = df_1d['high'].values[0] - df_1d['low'].values[0]  # first TR
    tr14 = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    hh14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(tr14 / ((hh14 - ll14) * 14)) / np.log10(10)  # simplify: log10(x)/log10(10) = log10(x)
    # Actually: CHOP = 100 * LOG10(sum(TR14)/(LOG10(HH14-LL14)*14)) -> we can compute directly
    chop_raw = 100 * np.log10(tr14 / ((hh14 - ll14) * 14)) / np.log10(10)
    # Fix: the above is wrong. Correct formula:
    # CHOP = 100 * (LOG10(sum(TR14)/14) / LOG10(HH14 - LL14))
    chop_raw = 100 * (np.log10(tr14 / 14) / np.log10(hh14 - ll14))
    # Handle division by zero and invalid values
    chop_raw = np.where((hh14 - ll14) > 0, chop_raw, 50.0)  # default to neutral
    
    # Align all 1d indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_raw)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_raw)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_raw)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian and indicators
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(vol_ma_aligned[i]) or np.isnan(chop_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        
        if vol_ma_aligned[i] <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (vol_ma_aligned[i] * 2.0)  # Volume spike threshold
        
        # Donchian breakout signals
        breakout_up = curr_high > donchian_high_aligned[i]  # break above Donchian high
        breakout_down = curr_low < donchian_low_aligned[i]  # break below Donchian low
        
        # Choppiness regime: >61.8 = range (mean reversion), <38.2 = trending
        chop_value = chop_aligned[i]
        in_range = chop_value > 61.8
        in_trend = chop_value < 38.2
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Donchian high AND volume confirmation AND in range (mean reversion long at lower band)
            if (breakout_up and 
                volume_confirm and 
                in_range):
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low AND volume confirmation AND in trend (trend continuation short)
            elif (breakout_down and 
                  volume_confirm and 
                  in_trend):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Donchian low (stoploss) OR chop turns trending (exit range)
            if (curr_low < donchian_low_aligned[i] or 
                not in_range):  # chop < 61.8 exits range regime
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high (stoploss) OR chop turns ranging (exit trend)
            if (curr_high > donchian_high_aligned[i] or 
                not in_trend):  # chop > 38.2 exits trend regime
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals