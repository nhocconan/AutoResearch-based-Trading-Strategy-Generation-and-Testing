#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and chop regime filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for Camarilla pivot levels (H3, L3) and volume average.
- Entry: Long when price breaks above H3 with volume > 1.5x 20-period average AND chop < 61.8 (trending).
         Short when price breaks below L3 with volume > 1.5x 20-period average AND chop < 61.8 (trending).
- Exit: Price returns to pivot point (PP) or opposite Camarilla level (L3 for long, H3 for short).
- Signal size: 0.25 discrete to minimize fee drag.
- Camarilla pivots identify intraday support/resistance; breakouts with volume indicate institutional participation.
- Chop regime filter ensures we only trade in trending markets, avoiding whipsaws in ranging conditions.
- Works in bull markets (buy breakouts) and bear markets (sell breakdowns).
- Estimated trades: ~100 total over 4 years (~25/year) based on Camarilla breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d HTF data for Camarilla pivots and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from 1d OHLC
    # Camarilla: PP = (H + L + C) / 3
    #          H3 = C + (H - L) * 1.1/4
    #          L3 = C - (H - L) * 1.1/4
    #          H4 = C + (H - L) * 1.1/2
    #          L4 = C - (H - L) * 1.1/2
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    pp = typical_price.values
    hl_range = (df_1d['high'] - df_1d['low']).values
    h3 = pp + hl_range * 1.1 / 4
    l3 = pp - hl_range * 1.1 / 4
    h4 = pp + hl_range * 1.1 / 2
    l4 = pp - hl_range * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (with 1-bar delay for completed 1d bar)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3, additional_delay_bars=1)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3, additional_delay_bars=1)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4, additional_delay_bars=1)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4, additional_delay_bars=1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp, additional_delay_bars=1)
    
    # Calculate 1d volume average (20-period) for confirmation
    vol_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20, additional_delay_bars=1)
    
    # Calculate chopiness index on 12h for regime filter (trending when < 38.2, ranging when > 61.8)
    # We'll use chop > 61.8 as ranging (avoid), chop < 38.2 as trending (favor)
    # Chop = 100 * log10(sum(ATR(14)) / log10(n) / (max(high) - min(low)))
    # Simplified: use rolling std dev of returns as proxy for chop
    returns = np.diff(np.log(close), prepend=0)
    chop = pd.Series(returns).rolling(window=14, min_periods=14).std() * np.sqrt(14) * 100
    chop_values = chop.values
    # Normalize chop to 0-100 scale (typical chop ranges 0-100)
    chop_values = np.clip(chop_values, 0, 100)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for volume MA and chop
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(chop_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: price returns to pivot point (PP) or opposite Camarilla level
        if position != 0:
            # Exit long: price falls below PP or rises above H4 (overbought)
            if position == 1:
                if curr_close < pp_aligned[i] or curr_close > h4_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price rises above PP or falls below L4 (oversold)
            elif position == -1:
                if curr_close > pp_aligned[i] or curr_close < l4_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volume confirmation and trending regime
        if position == 0:
            # Volume confirmation: current volume > 1.5x 20-period average
            volume_confirm = curr_volume > 1.5 * vol_ma_20_aligned[i]
            # Regime filter: chop < 50 (favor trending over ranging)
            trending_regime = chop_values[i] < 50
            
            # Long: price breaks above H3 with volume confirmation and trending regime
            if curr_close > h3_aligned[i] and volume_confirm and trending_regime:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 with volume confirmation and trending regime
            elif curr_close < l3_aligned[i] and volume_confirm and trending_regime:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dVolMa20_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0