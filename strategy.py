#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1w volume spike + 1d choppiness regime filter
# - Long when price breaks above Donchian(20) high + 1w volume > 2.0x 20-period volume average + 1d chop > 61.8 (ranging market for mean reversion failure avoidance)
# - Short when price breaks below Donchian(20) low + 1w volume > 2.0x 20-period volume average + 1d chop > 61.8
# - Exit when price returns to Donchian(20) midpoint or ATR stoploss triggered (adverse move > 2.0*ATR)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Works in bull/bear: Donchian captures breakouts; volume confirms participation; chop filter avoids whipsaws in strong trends
# - Target: 15-35 trades/year to stay within fee drag limits while capturing strong moves

name = "12h_1w_1d_donchian_volume_chop_v1"
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
    entry_price = 0.0
    
    # Load 12h data ONCE before loop for Donchian and ATR (MTF rule compliance)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return signals
    
    # Load 1w data ONCE before loop for volume confirmation (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Load 1d data ONCE before loop for choppiness regime filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 12h Donchian(20) channels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high_20 + lowest_low_20) / 2.0
    
    # Pre-compute 12h ATR(20) for stoploss
    tr1 = pd.Series(high).rolling(2).max() - pd.Series(low).rolling(2).min()
    tr2 = abs(pd.Series(high).shift(1) - pd.Series(close))
    tr3 = abs(pd.Series(low).shift(1) - pd.Series(close))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_20 = tr.rolling(window=20, min_periods=20).mean().values
    atr_20_aligned = align_htf_to_ltf(prices, df_12h, atr_20)
    
    # Pre-compute 1w volume SMA for confirmation
    volume_1w = df_1w['volume'].values
    volume_sma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_sma_20_1w)
    
    # Pre-compute 1d choppiness index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(highest_high - lowest_low) * sqrt(14)))
    # Simplified: CHOP = 100 * log10(ATR_sum / (log10(HH-LL) * sqrt(14))) but we use standard formula
    atr_14 = pd.Series(high).rolling(2).max() - pd.Series(low).rolling(2).min()
    atr_14 = atr_14.rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid log10(0) and division by zero
    hh_ll = highest_high_14 - lowest_low_14
    hh_ll = np.where(hh_ll <= 0, 1e-10, hh_ll)
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(atr_sum) / (np.log10(hh_ll) * np.sqrt(14))
    chop = np.where(np.isnan(chop) | np.isinf(chop), 50, chop)  # default to neutral
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(atr_20_aligned[i]) or np.isnan(volume_sma_20_1w_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        
        # Volume confirmation: 1w volume > 2.0x 20-period volume average (strict threshold)
        volume_1w_current = df_1w['volume'].values
        volume_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w_current)
        vol_confirm = volume_1w_aligned[i] > 2.0 * volume_sma_20_1w_aligned[i]
        
        # Choppiness regime filter: CHOP > 61.8 indicates ranging market (avoid strong trends)
        chop_condition = chop_aligned[i] > 61.8
        
        # Donchian breakout conditions
        donchian_breakout_up = close_price > highest_high_20[i-1]  # Use previous bar's channel
        donchian_breakout_down = close_price < lowest_low_20[i-1]  # Use previous bar's channel
        
        # Entry conditions
        enter_long = donchian_breakout_up and vol_confirm and chop_condition
        enter_short = donchian_breakout_down and vol_confirm and chop_condition
        
        # Exit conditions
        exit_long = (position == 1 and 
                    (close_price < donchian_mid[i] or  # Return to midpoint
                     close_price < entry_price - 2.0 * atr_20_aligned[i]))  # ATR stoploss
        exit_short = (position == -1 and 
                     (close_price > donchian_mid[i] or  # Return to midpoint
                      close_price > entry_price + 2.0 * atr_20_aligned[i]))  # ATR stoploss
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            entry_price = close_price
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            entry_price = close_price
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            entry_price = 0.0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            entry_price = 0.0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals