#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout + 1d chop regime + session filter
# - Uses 4h Donchian(20) for signal direction (trend/range structure)
# - Uses 1d Chop (>61.8) to confirm ranging market where breakouts fade
# - Uses 1h session filter (08-20 UTC) to avoid low-liquidity periods
# - Entry timing on 1h: wait for pullback to Donchian middle after breakout
# - Discrete position sizing 0.20 to minimize fee churn
# - Target: 15-37 trades/year (60-150 total over 4 years) on 1h timeframe
# - Works in both bull/bear via mean reversion at Donchian extremes in ranging markets

name = "1h_4h_1d_donchian_chop_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Pre-compute 1h data arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 4h Donchian channels (20) - using 4h data aligned to 1h
    df_4h_close = df_4h['close'].values
    df_4h_high = df_4h['high'].values
    df_4h_low = df_4h['low'].values
    
    # Calculate Donchian on 4h, then align to 1h
    donchian_high_4h = pd.Series(df_4h_high).rolling(window=20, min_periods=20).max().values
    donchian_low_4h = pd.Series(df_4h_low).rolling(window=20, min_periods=20).min().values
    donchian_middle_4h = (donchian_high_4h + donchian_low_4h) / 2
    
    # Align 4h Donchian levels to 1h timeframe
    donchian_high = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    donchian_middle = align_htf_to_ltf(prices, df_4h, donchian_middle_4h)
    
    # Pre-compute 1h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1d chop regime (choppiness index)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - np.roll(close_1d, 1)[1:])
    tr3 = np.abs(low_1d[1:] - np.roll(close_1d, 1)[1:])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # first element is NaN
    
    # ATR(14)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    
    # Max(high) - Min(low) over 14 periods
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_max_min = max_high - min_low
    
    # Chop = 100 * log10(tr_sum / range_max_min) / log10(14)
    chop = 100 * np.log10(tr_sum / range_max_min) / np.log10(14)
    chop = np.concatenate([np.full(13, np.nan), chop[13:]])  # align indices
    
    # Chop regime: > 61.8 = ranging (good for mean reversion at extremes)
    chop_range = chop > 61.8
    
    # Align 1d chop regime to 1h timeframe
    chop_range_aligned = align_htf_to_ltf(prices, df_1d, chop_range)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid or outside session
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop_range_aligned[i]) or
            not session_filter[i]):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above 4h Donchian high AND volume spike AND chop range
            # Entry on pullback to middle after breakout confirmation
            if (close[i] > donchian_high[i-1] and  # breakout above previous 4h high
                close[i] <= donchian_middle[i] and  # pulled back to middle
                volume_spike[i] and 
                chop_range_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short conditions: price breaks below 4h Donchian low AND volume spike AND chop range
            elif (close[i] < donchian_low[i-1] and  # breakout below previous 4h low
                  close[i] >= donchian_middle[i] and  # pulled back to middle
                  volume_spike[i] and 
                  chop_range_aligned[i]):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price returns to opposite Donchian level or signals reverse
            exit_long = (position == 1 and 
                        (close[i] <= donchian_low[i] or  # reached opposite extreme
                         close[i] >= donchian_high[i]))  # or back to breakout level
            exit_short = (position == -1 and 
                         (close[i] >= donchian_high[i] or  # reached opposite extreme
                          close[i] <= donchian_low[i]))   # or back to breakout level
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
    
    return signals