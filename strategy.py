#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 12h volume spike + chop regime filter
# - Primary: 6h Donchian breakout (20-period) for trend continuation
# - HTF: 12h volume confirmation (current volume > 2.0x 24-period MA) for conviction
# - Regime: 6h choppy market filter (CHOP(14) > 61.8 = avoid breakouts in ranging markets)
# - Long: Price breaks above Donchian upper band + volume confirmation + chop < 61.8
# - Short: Price breaks below Donchian lower band + volume confirmation + chop < 61.8
# - Exit: Price crosses Donchian middle band (median of upper/lower) or opposite breakout
# - Position sizing: 0.25 (discrete level to balance return and drawdown)
# - Works in bull/bear: Donchian adapts to volatility, volume filters false breakouts, chop regime avoids whipsaws
# - Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_12h_donchian_volume_chop_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough data for volume MA
        return np.zeros(n)
    
    # Pre-compute 6h data
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    
    # Pre-compute 12h data for volume confirmation
    volume_12h = df_12h['volume'].values
    
    # Calculate 6h Donchian channels (20-period) - use previous bars to avoid look-ahead
    donchian_upper = np.full(len(close_6h), np.nan)
    donchian_lower = np.full(len(close_6h), np.nan)
    donchian_middle = np.full(len(close_6h), np.nan)
    
    for i in range(20, len(close_6h)):
        if not (np.isnan(high_6h[i-20:i]).any() or np.isnan(low_6h[i-20:i]).any()):
            donchian_upper[i] = np.max(high_6h[i-20:i])
            donchian_lower[i] = np.min(low_6h[i-20:i])
            donchian_middle[i] = (donchian_upper[i] + donchian_lower[i]) / 2.0
    
    # Calculate 12h volume moving average (24-period) for volume confirmation
    volume_ma_24_12h = np.full(len(volume_12h), np.nan)
    for i in range(23, len(volume_12h)):
        if not np.isnan(volume_12h[i-23:i+1]).any():
            volume_ma_24_12h[i] = np.mean(volume_12h[i-23:i+1])
    
    # Calculate 6h Choppiness Index (CHOP) - 14 period
    chop = np.full(len(close_6h), np.nan)
    atr_14 = np.full(len(close_6h), np.nan)
    
    # First calculate True Range and ATR(14)
    tr = np.full(len(close_6h), np.nan)
    for i in range(1, len(close_6h)):
        if not (np.isnan(high_6h[i]) or np.isnan(low_6h[i]) or np.isnan(close_6h[i-1])):
            tr[i] = max(
                high_6h[i] - low_6h[i],
                abs(high_6h[i] - close_6h[i-1]),
                abs(low_6h[i] - close_6h[i-1])
            )
    
    # Calculate ATR(14) using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    for i in range(14, len(tr)):
        if not np.isnan(tr[i-13:i+1]).any():
            if i == 14:
                atr_14[i] = np.mean(tr[1:15])  # First ATR is simple average
            else:
                atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Calculate Choppiness Index: CHOP = 100 * log10(sum(ATR14)/log10(n)) / log10(n)
    for i in range(27, len(close_6h)):  # Need 14 ATR + 14 period for CHOP
        if not np.isnan(atr_14[i-13:i+1]).any():
            sum_atr = np.sum(atr_14[i-13:i+1])
            if sum_atr > 0 and close_6h[i] > 0:
                max_high = np.max(high_6h[i-13:i+1])
                min_low = np.min(low_6h[i-13:i+1])
                if max_high > min_low:
                    chop[i] = 100 * np.log10(sum_atr) / np.log10(14) / np.log10(max_high - min_low)
    
    # Align all HTF indicators to 6h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, prices, donchian_upper)  # Same timeframe
    donchian_lower_aligned = align_htf_to_ltf(prices, prices, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, prices, donchian_middle)
    chop_aligned = align_htf_to_ltf(prices, prices, chop)
    volume_ma_24_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_24_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(28, n):  # Start after warmup period for all indicators
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ma_24_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 2.0x 24-period MA
        volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        volume_confirm = volume_12h_aligned[i] > 2.0 * volume_ma_24_12h_aligned[i]
        
        # Chop regime filter: only trade when market is trending (CHOP < 61.8)
        trending_regime = chop_aligned[i] < 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian upper + volume confirmation + trending regime
            if close_6h[i] > donchian_upper_aligned[i] and volume_confirm and trending_regime:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian lower + volume confirmation + trending regime
            elif close_6h[i] < donchian_lower_aligned[i] and volume_confirm and trending_regime:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price crosses Donchian middle band OR opposite breakout
            if position == 1:  # Long position
                if close_6h[i] < donchian_middle_aligned[i] or close_6h[i] < donchian_lower_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close_6h[i] > donchian_middle_aligned[i] or close_6h[i] > donchian_upper_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals