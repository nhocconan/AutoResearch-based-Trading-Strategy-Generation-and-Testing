#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout + 1d volume spike + chop regime filter
# - Primary: 12h price breaks above/below Camarilla pivot levels (H3/L3) from prior 1d
# - HTF: 1d volume confirmation (current volume > 1.8x 20-period MA) for conviction
# - Regime: 12h choppy market filter (CHOP(14) > 61.8 = avoid signals in ranging markets)
# - Long: Close > H3 (1.1/1.2) + volume confirmation + chop < 61.8
# - Short: Close < L3 (1.1/1.2) + volume confirmation + chop < 61.8
# - Exit: Close crosses back inside H3/L3 levels OR chop > 61.8 (regime change to ranging)
# - Position sizing: 0.25 (discrete level to balance return and drawdown)
# - Works in bull/bear: Camarilla levels adapt to volatility, volume filters false signals, chop regime avoids whipsaws
# - Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

name = "12h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:  # Need enough data for calculations
        return np.zeros(n)
    
    # Pre-compute 12h data
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla pivot levels (H3, L3, H4, L4)
    # Based on prior day's high, low, close
    camarilla_h3 = np.full(len(close_1d), np.nan)
    camarilla_l3 = np.full(len(close_1d), np.nan)
    camarilla_h4 = np.full(len(close_1d), np.nan)
    camarilla_l4 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        if not (np.isnan(high_1d[i-1]) or np.isnan(low_1d[i-1]) or np.isnan(close_1d[i-1])):
            # Calculate pivot point
            pp = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3.0
            range_ = high_1d[i-1] - low_1d[i-1]
            
            # Camarilla levels
            camarilla_h3[i] = pp + range_ * 1.1 / 4.0
            camarilla_l3[i] = pp - range_ * 1.1 / 4.0
            camarilla_h4[i] = pp + range_ * 1.2 / 4.0
            camarilla_l4[i] = pp - range_ * 1.2 / 4.0
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        if not np.isnan(volume_1d[i-19:i+1]).any():
            volume_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Calculate 12h Choppiness Index (CHOP) - 14 period
    chop = np.full(len(close_12h), np.nan)
    atr_14 = np.full(len(close_12h), np.nan)
    
    # First calculate True Range and ATR(14)
    tr = np.full(len(close_12h), np.nan)
    for i in range(1, len(close_12h)):
        if not (np.isnan(high_12h[i]) or np.isnan(low_12h[i]) or np.isnan(close_12h[i-1])):
            tr[i] = max(
                high_12h[i] - low_12h[i],
                abs(high_12h[i] - close_12h[i-1]),
                abs(low_12h[i] - close_12h[i-1])
            )
    
    # Calculate ATR(14) using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    for i in range(14, len(tr)):
        if not np.isnan(tr[i-13:i+1]).any():
            if i == 14:
                atr_14[i] = np.mean(tr[1:15])  # First ATR is simple average
            else:
                atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Calculate Choppiness Index: CHOP = 100 * log10(sum(ATR14)) / log10(14) / log10(max_high - min_low)
    for i in range(27, len(close_12h)):  # Need 14 ATR + 14 period for CHOP
        if not np.isnan(atr_14[i-13:i+1]).any():
            sum_atr = np.sum(atr_14[i-13:i+1])
            if sum_atr > 0:
                max_high = np.max(high_12h[i-13:i+1])
                min_low = np.min(low_12h[i-13:i+1])
                if max_high > min_low:
                    chop[i] = 100 * np.log10(sum_atr) / np.log10(14) / np.log10(max_high - min_low)
    
    # Align all HTF indicators to 12h timeframe
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
            np.isnan(chop_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.8x 20-period MA
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirm = volume_1d_aligned[i] > 1.8 * volume_ma_20_1d_aligned[i]
        
        # Chop regime filter: only trade when market is trending (CHOP < 61.8)
        trending_regime = chop_aligned[i] < 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Close > H3 (1.1) + volume confirmation + trending regime
            if close_12h[i] > camarilla_h3_aligned[i] and volume_confirm and trending_regime:
                position = 1
                signals[i] = 0.25
            # Short entry: Close < L3 (1.1) + volume confirmation + trending regime
            elif close_12h[i] < camarilla_l3_aligned[i] and volume_confirm and trending_regime:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Close crosses back inside H3/L3 levels OR chop > 61.8 (regime change to ranging)
            if position == 1:  # Long position
                if close_12h[i] < camarilla_h3_aligned[i] or chop_aligned[i] >= 61.8:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close_12h[i] > camarilla_l3_aligned[i] or chop_aligned[i] >= 61.8:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals