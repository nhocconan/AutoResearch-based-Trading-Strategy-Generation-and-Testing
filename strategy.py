#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and choppiness regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 2.5x 20-bar avg AND CHOP(14) < 61.8 (trending)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 2.5x 20-bar avg AND CHOP(14) < 61.8 (trending)
# - Exit when price returns to Camarilla pivot point (mean reversion to equilibrium)
# - Uses 1d volume for confirmation to ensure institutional participation
# - Uses choppiness regime filter to avoid false breakouts in ranging markets
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - Camarilla pivots identify key intraday support/resistance levels from prior day
# - Breakouts with volume confirmation capture strong institutional moves
# - Choppiness filter ensures we only trade in trending environments, reducing whipsaws

name = "4h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Camarilla pivot levels from prior 1d bar
    # Camarilla levels: H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low)
    #                 L3 = close - 1.1*(high-low), L4 = close - 1.5*(high-low)
    #                 Pivot = (high + low + close)/3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Pre-compute 1d volume confirmation: > 2.5x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (2.5 * volume_20_avg)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute choppiness index on 4h data for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high,14) - min(low,14))) / log10(14)
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    high_14 = prices['high'].rolling(window=14, min_periods=14).max().values
    low_14 = prices['low'].rolling(window=14, min_periods=14).min().values
    
    # True Range components
    tr1 = prices['high'] - prices['low']
    tr2 = abs(prices['high'] - prices['close'].shift(1))
    tr3 = abs(prices['low'] - prices['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chopiness calculation
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    high_low_range = high_14 - low_14
    chop = np.where(
        (high_low_range > 0) & ~np.isnan(atr_sum),
        100 * np.log10(atr_sum / high_low_range) / np.log10(14),
        50.0  # neutral when undefined
    )
    # Trending regime: CHOP < 61.8 (below upper ranging threshold)
    trending_regime = chop < 61.8
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_pivot_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(chop[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Camarilla H3 AND volume spike AND trending regime
            if (prices['close'].iloc[i] > camarilla_h3_aligned[i] and 
                vol_spike_1d_aligned.iloc[i] and 
                trending_regime[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Camarilla L3 AND volume spike AND trending regime
            elif (prices['close'].iloc[i] < camarilla_l3_aligned[i] and 
                  vol_spike_1d_aligned.iloc[i] and 
                  trending_regime[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to Camarilla pivot (mean reversion)
            # Exit when price returns to Camarilla pivot point
            exit_signal = False
            if position == 1:  # Long position
                if prices['close'].iloc[i] <= camarilla_pivot_aligned[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['close'].iloc[i] >= camarilla_pivot_aligned[i]:
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