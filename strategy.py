#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and choppiness regime filter
# - Long when price breaks above Camarilla H3 (1d) AND 1d volume > 2.0x 20-bar avg AND choppiness < 61.8 (trending)
# - Short when price breaks below Camarilla L3 (1d) AND 1d volume > 2.0x 20-bar avg AND choppiness < 61.8
# - Exit when price returns to Camarilla pivot point (1d) or opposite L3/H3 level
# - Uses discrete position sizing (0.25) to balance return and drawdown
# - Camarilla levels provide precise intraday support/resistance from prior day action
# - Volume confirmation ensures breakout legitimacy
# - Choppiness filter avoids whipsaws in ranging markets
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)

name = "4h_1d_camarilla_breakout_volume_chop_v1"
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
    
    # Pre-compute 1d Camarilla pivot levels from prior day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations: based on prior day's range
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3
    camarilla_range = high_1d - low_1d
    camarilla_h3 = camarilla_pivot + (camarilla_range * 1.1 / 4)
    camarilla_l3 = camarilla_pivot - (camarilla_range * 1.1 / 4)
    camarilla_h4 = camarilla_pivot + (camarilla_range * 1.1 / 2)
    camarilla_l4 = camarilla_pivot - (camarilla_range * 1.1 / 2)
    
    # Pre-compute 1d volume confirmation: > 2.0x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (2.0 * volume_20_avg_1d)
    
    # Pre-compute 1d choppiness index: CHOP(14) < 61.8 = trending (favor breakouts)
    # CHOP = 100 * log10(SUM(ATR(1),14) / (log10(n) * (MAX(high,14) - MIN(low,14))))
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr_1d = np.concatenate([[np.nan], tr_1d])  # align length
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denominator = np.log10(14) * (max_high_14 - min_low_14)
    chop_1d = np.where(
        (chop_denominator != 0) & (~np.isnan(chop_denominator)),
        100 * np.log10(sum_atr_14 / chop_denominator),
        50  # default when undefined
    )
    chop_trending = chop_1d < 61.8  # trending market
    
    # Align HTF indicators to 4h timeframe
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    chop_trending_aligned = align_htf_to_ltf(prices, df_1d, chop_trending)
    
    # Pre-compute session filter: 08-20 UTC (avoid low-volume Asian session)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_pivot_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or
            np.isnan(chop_trending_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Apply session filter
        if not in_session[i]:
            # Outside session: flatten position
            position = 0
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H3 AND volume spike AND trending market
            if (prices['close'].iloc[i] > camarilla_h3_aligned[i] and
                vol_spike_1d_aligned[i] and
                chop_trending_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below L3 AND volume spike AND trending market
            elif (prices['close'].iloc[i] < camarilla_l3_aligned[i] and
                  vol_spike_1d_aligned[i] and
                  chop_trending_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price returns to pivot point or opposite level
            exit_long = (prices['close'].iloc[i] <= camarilla_pivot_aligned[i]) or \
                       (prices['close'].iloc[i] >= camarilla_h4_aligned[i])
            exit_short = (prices['close'].iloc[i] >= camarilla_pivot_aligned[i]) or \
                        (prices['close'].iloc[i] <= camarilla_l4_aligned[i])
            
            if (position == 1 and exit_long) or (position == -1 and exit_short):
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals