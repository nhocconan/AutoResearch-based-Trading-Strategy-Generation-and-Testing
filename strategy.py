#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation + choppiness filter
# - Long when price breaks above Donchian upper (20) on 4h AND 1d volume > 1.5x 20-bar avg AND 1d choppiness < 61.8 (trending)
# - Short when price breaks below Donchian lower (20) on 4h AND 1d volume > 1.5x 20-bar avg AND 1d choppiness < 61.8
# - Exit when price crosses Donchian midpoint (mean reversion structure)
# - Uses discrete position sizing (0.30) for optimal risk/return
# - Donchian provides clear structure, volume confirms participation, chop filter avoids false breakouts in ranges
# - Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
# - Works in bull markets (breakouts) and bear markets (breakdowns) with trend filter

name = "4h_1d_donchian_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * volume_20_avg_1d)
    
    # Pre-compute 1d choppiness index: CHOP(14) < 61.8 = trending (favor breakouts)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of true range over 14 periods
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness index: 100 * log10(tr_sum_14 / (atr_14 * 14)) / log10(14)
    chop_denom = atr_14 * 14
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid div by zero
    chop_ratio = tr_sum_14 / chop_denom
    chop_ratio = np.where(chop_ratio <= 0, 1e-10, chop_ratio)  # avoid log(<=0)
    chop_1d = 100 * np.log10(chop_ratio) / np.log10(14)
    chop_trending_1d = chop_1d < 61.8  # trending market
    
    # Pre-compute 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align HTF indicators to 4h timeframe
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    chop_trending_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_trending_1d)
    
    # Session filter: 08-20 UTC (avoid low liquidity Asian session)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(vol_spike_1d_aligned[i]) or np.isnan(chop_trending_1d_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        # Apply session filter
        if not in_session[i]:
            # Outside session: flatten position
            position = 0
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian high AND 1d volume spike AND 1d trending
            if (prices['high'].iloc[i] > donchian_high[i] and 
                vol_spike_1d_aligned[i] and 
                chop_trending_1d_aligned[i]):
                position = 1
                signals[i] = 0.30
            # Short when price breaks below Donchian low AND 1d volume spike AND 1d trending
            elif (prices['low'].iloc[i] < donchian_low[i] and 
                  vol_spike_1d_aligned[i] and 
                  chop_trending_1d_aligned[i]):
                position = -1
                signals[i] = -0.30
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to Donchian midpoint
            # Exit when price crosses Donchian midpoint
            exit_long = position == 1 and prices['close'].iloc[i] <= donchian_mid[i]
            exit_short = position == -1 and prices['close'].iloc[i] >= donchian_mid[i]
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.30
                else:
                    signals[i] = -0.30
    
    return signals