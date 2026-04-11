#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout + 1d volume spike + 1w choppiness regime filter
# - Long: price breaks above Camarilla H3 level + 1d volume > 1.3x 20-period volume average + 1w Choppiness Index > 61.8 (ranging market)
# - Short: price breaks below Camarilla L3 level + 1d volume > 1.3x 20-period volume average + 1w Choppiness Index > 61.8 (ranging market)
# - Exit: price reverses back to Camarilla H4/L4 levels or opposite pivot touch
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 20-50 trades/year to stay within fee drag limits
# - Choppiness filter ensures we only trade in ranging markets where mean reversion works
# - Works in both bull and bear markets by focusing on ranging conditions where price respects pivot levels

name = "4h_1d_1w_camarilla_chop_v1"
timeframe = "4h"
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
    
    # Load 4h data ONCE before loop for Camarilla calculation (MTF rule compliance)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return signals
    
    # Load 1d data ONCE before loop for volume confirmation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Load 1w data ONCE before loop for Choppiness Index (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute 4h Camarilla levels (based on previous day's range)
    # Camarilla uses previous period's high, low, close
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    prev_close = df_4h['close'].shift(1).values
    
    # Calculate Camarilla levels for 4h timeframe
    # H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    # H4 = C + (H-L)*1.1/2, L4 = C - (H-L)*1.1/2
    camarilla_h3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_l3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    camarilla_h4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_l4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l4)
    
    # Pre-compute 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute 1w Choppiness Index (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1_w = pd.Series(high_1w).rolling(2).max() - pd.Series(low_1w).rolling(2).min()
    tr2_w = abs(pd.Series(high_1w).shift(1) - pd.Series(close_1w))
    tr3_w = abs(pd.Series(low_1w).shift(1) - pd.Series(close_1w))
    tr_w = pd.concat([tr1_w, tr2_w, tr3_w], axis=1).max(axis=1)
    atr_w = tr_w.rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index = 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
    # We'll use a simplified version: CHOP = 100 * log10(ATR14_sum / (HHV - LLV)) / log10(period)
    # Higher CHOP (>61.8) = ranging market, Lower CHOP (<38.2) = trending market
    atr_sum_14 = pd.Series(atr_w).rolling(window=14, min_periods=14).sum().values
    hhvl_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    llvl_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    chop_numerator = atr_sum_14 / (hhvl_1w - llvl_1w + 1e-10)
    chop_numerator = np.where(chop_numerator > 0, chop_numerator, 1e-10)
    chop = 100 * np.log10(chop_numerator) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        
        # Volume confirmation: 1d volume > 1.3x 20-period volume average (moderate threshold)
        volume_1d_current = df_1d['volume'].values
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_current)
        vol_confirm = volume_1d_aligned[i] > 1.3 * volume_sma_20_1d_aligned[i]
        
        # Weekly chop filter: Choppiness Index > 61.8 (ranging market)
        weekly_chop = chop_aligned[i]
        chop_filter = weekly_chop > 61.8
        
        # Camarilla breakout conditions
        camarilla_breakout_long = close_price > camarilla_h3_aligned[i]
        camarilla_breakout_short = close_price < camarilla_l3_aligned[i]
        
        # Entry conditions
        enter_long = camarilla_breakout_long and vol_confirm and chop_filter
        enter_short = camarilla_breakout_short and vol_confirm and chop_filter
        
        # Exit conditions: price reverses to H4/L4 levels or touches opposite pivot
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price drops below H4 or touches L3 (mean reversion)
            exit_long = close_price < camarilla_h4_aligned[i] or low_price <= camarilla_l3_aligned[i]
        elif position == -1:
            # Exit short if price rises above L4 or touches H3 (mean reversion)
            exit_short = close_price > camarilla_l4_aligned[i] or high_price >= camarilla_h3_aligned[i]
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals