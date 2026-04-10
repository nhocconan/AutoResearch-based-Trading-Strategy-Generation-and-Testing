#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout + 1d volume spike + chop regime filter
# - Primary: 4h Camarilla pivot levels from previous 1d (H3/L3) for institutional breakout
# - HTF: 1d volume confirmation (current volume > 2.0x 20-period MA) for conviction
# - Regime: 4h choppy market filter (CHOP(14) > 61.8 = avoid breakouts in ranging markets)
# - Long: Price breaks above Camarilla H3 + volume confirmation + chop < 61.8
# - Short: Price breaks below Camarilla L3 + volume confirmation + chop < 61.8
# - Exit: Price crosses Camarilla H4/L4 levels or opposite pivot breakout
# - Position sizing: 0.25 (discrete level to balance return and drawdown)
# - Works in bull/bear: Camarilla adapts to volatility, volume filters false breakouts, chop regime avoids whipsaws
# - Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_1d_camarilla_volume_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:  # Need enough data for calculations
        return np.zeros(n)
    
    # Pre-compute 4h data
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    camarilla_h4 = np.full(len(close_1d), np.nan)
    camarilla_h3 = np.full(len(close_1d), np.nan)
    camarilla_l3 = np.full(len(close_1d), np.nan)
    camarilla_l4 = np.full(len(close_1d), np.nan)
    camarilla_h5 = np.full(len(close_1d), np.nan)
    camarilla_l5 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):  # Start from 1 to use previous day
        if not (np.isnan(high_1d[i-1]) or np.isnan(low_1d[i-1]) or np.isnan(close_1d[i-1])):
            range_prev = high_1d[i-1] - low_1d[i-1]
            camarilla_h4[i] = close_1d[i-1] + range_prev * 1.1 / 2
            camarilla_h3[i] = close_1d[i-1] + range_prev * 1.1 / 4
            camarilla_l3[i] = close_1d[i-1] - range_prev * 1.1 / 4
            camarilla_l4[i] = close_1d[i-1] - range_prev * 1.1 / 2
            camarilla_h5[i] = close_1d[i-1] + range_prev * 1.1 * 2 / 2
            camarilla_l5[i] = close_1d[i-1] - range_prev * 1.1 * 2 / 2
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        if not np.isnan(volume_1d[i-19:i+1]).any():
            volume_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Calculate 4h Choppiness Index (CHOP) - 14 period
    chop = np.full(len(close_4h), np.nan)
    atr_14 = np.full(len(close_4h), np.nan)
    
    # First calculate True Range and ATR(14)
    tr = np.full(len(close_4h), np.nan)
    for i in range(1, len(close_4h)):
        if not (np.isnan(high_4h[i]) or np.isnan(low_4h[i]) or np.isnan(close_4h[i-1])):
            tr[i] = max(
                high_4h[i] - low_4h[i],
                abs(high_4h[i] - close_4h[i-1]),
                abs(low_4h[i] - close_4h[i-1])
            )
    
    # Calculate ATR(14) using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    for i in range(14, len(tr)):
        if not np.isnan(tr[i-13:i+1]).any():
            if i == 14:
                atr_14[i] = np.mean(tr[1:15])  # First ATR is simple average
            else:
                atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Calculate Choppiness Index: CHOP = 100 * log10(sum(ATR14)) / log10(14) / log10(max_high - min_low)
    for i in range(27, len(close_4h)):  # Need 14 ATR + 14 period for CHOP
        if not np.isnan(atr_14[i-13:i+1]).any():
            sum_atr = np.sum(atr_14[i-13:i+1])
            if sum_atr > 0:
                max_high = np.max(high_4h[i-13:i+1])
                min_low = np.min(low_4h[i-13:i+1])
                if max_high > min_low:
                    chop[i] = 100 * np.log10(sum_atr) / np.log10(14) / np.log10(max_high - min_low)
    
    # Align all HTF indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    chop_aligned = align_htf_to_ltf(prices, prices, chop)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(28, n):  # Start after warmup period for all indicators
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x 20-period MA
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirm = volume_1d_aligned[i] > 2.0 * volume_ma_20_1d_aligned[i]
        
        # Chop regime filter: only trade when market is trending (CHOP < 61.8)
        trending_regime = chop_aligned[i] < 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Camarilla H3 + volume confirmation + trending regime
            if close_4h[i] > camarilla_h3_aligned[i] and volume_confirm and trending_regime:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Camarilla L3 + volume confirmation + trending regime
            elif close_4h[i] < camarilla_l3_aligned[i] and volume_confirm and trending_regime:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price crosses Camarilla H4/L4 levels OR opposite pivot breakout
            if position == 1:  # Long position
                if close_4h[i] < camarilla_l4_aligned[i] or close_4h[i] < camarilla_l3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close_4h[i] > camarilla_h4_aligned[i] or close_4h[i] > camarilla_h3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals