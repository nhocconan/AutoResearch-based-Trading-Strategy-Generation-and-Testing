#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout/mean reversion + 12h volume confirmation + 1d ADX regime filter
# - Primary: 6h price breaks above Camarilla R4 or below S4 (strong breakout)
# - Secondary: 6h price reverts to Camarilla H3/L3 levels (mean reversion in range)
# - HTF: 12h volume > 1.5x 20-period MA for confirmation (avoids low-volume false signals)
# - Regime filter: 1d ADX(14) > 25 = trending market (use breakout logic), ADX < 20 = ranging (use mean reversion)
# - Long breakout: Price > R4 + volume confirmation + ADX > 25
# - Short breakout: Price < S4 + volume confirmation + ADX > 25
# - Long mean reversion: Price < L3 + ADX < 20 (oversold in range)
# - Short mean reversion: Price > H3 + ADX < 20 (overbought in range)
# - Exit: Price reaches opposite Camarilla level (H3 for L3 longs, L3 for H3 shorts) or midpoint for breakouts
# - Position sizing: 0.25 (discrete level, balances return/drawdown, reduces fee churn)
# - Works in bull/bear: ADX regime adapts strategy to market conditions, volume filters false signals
# - Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_12h_1d_camarilla_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 25 or len(df_1d) < 25:  # Need enough data for calculations
        return np.zeros(n)
    
    # Pre-compute primary (6h) data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute 12h data
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 6h Camarilla pivot levels (based on previous bar)
    camarilla_h3 = np.full(len(close), np.nan)
    camarilla_l3 = np.full(len(close), np.nan)
    camarilla_h4 = np.full(len(close), np.nan)
    camarilla_l4 = np.full(len(close), np.nan)
    camarilla_h3_ub = np.full(len(close), np.nan)  # Upper bound for H3 mean reversion
    camarilla_l3_lb = np.full(len(close), np.nan)  # Lower bound for L3 mean reversion
    
    for i in range(1, len(close)):
        if not (np.isnan(high[i-1]) or np.isnan(low[i-1]) or np.isnan(close[i-1])):
            # True range for Camarilla calculation
            tr = max(
                high[i-1] - low[i-1],
                abs(high[i-1] - close[i-1]),
                abs(low[i-1] - close[i-1])
            )
            camarilla_h3[i] = close[i-1] + 1.1 * tr / 6
            camarilla_l3[i] = close[i-1] - 1.1 * tr / 6
            camarilla_h4[i] = close[i-1] + 1.1 * tr / 2
            camarilla_l4[i] = close[i-1] - 1.1 * tr / 2
            camarilla_h3_ub[i] = camarilla_h3[i]  # Exit long mean reversion at H3
            camarilla_l3_lb[i] = camarilla_l3[i]  # Exit short mean reversion at L3
    
    # Calculate 12h volume moving average (20-period)
    volume_ma_20_12h = np.full(len(volume_12h), np.nan)
    for i in range(19, len(volume_12h)):
        if not np.isnan(volume_12h[i-19:i+1]).any():
            volume_ma_20_12h[i] = np.mean(volume_12h[i-19:i+1])
    
    # Calculate 1d ADX(14)
    adx = np.full(len(close_1d), np.nan)
    
    # True Range
    tr_1d = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i-1])):
            tr_1d[i] = max(
                high_1d[i] - low_1d[i],
                abs(high_1d[i] - close_1d[i-1]),
                abs(low_1d[i] - close_1d[i-1])
            )
    
    # Directional Movement
    plus_dm = np.full(len(close_1d), np.nan)
    minus_dm = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(high_1d[i-1]) or 
                np.isnan(low_1d[i]) or np.isnan(low_1d[i-1])):
            up_move = high_1d[i] - high_1d[i-1]
            down_move = low_1d[i-1] - low_1d[i]
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            elif down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
    
    # Smoothed ATR, +DM, -DM (using Wilder's smoothing)
    atr_1d = np.full(len(close_1d), np.nan)
    plus_dm_smooth = np.full(len(close_1d), np.nan)
    minus_dm_smooth = np.full(len(close_1d), np.nan)
    
    # Initial values (first 14 periods)
    for i in range(14, len(close_1d)):
        if not (np.isnan(tr_1d[i-13:i+1]).any() or 
                np.isnan(plus_dm[i-13:i+1]).any() or 
                np.isnan(minus_dm[i-13:i+1]).any()):
            atr_1d[i] = np.sum(tr_1d[i-13:i+1])
            plus_dm_smooth[i] = np.sum(plus_dm[i-13:i+1])
            minus_dm_smooth[i] = np.sum(minus_dm[i-13:i+1])
            break
    
    # Wilder's smoothing for subsequent periods
    for i in range(15, len(close_1d)):
        if not (np.isnan(atr_1d[i-1]) or np.isnan(tr_1d[i]) or
                np.isnan(plus_dm_smooth[i-1]) or np.isnan(plus_dm[i]) or
                np.isnan(minus_dm_smooth[i-1]) or np.isnan(minus_dm[i])):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * 13 + plus_dm[i]) / 14
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * 13 + minus_dm[i]) / 14
    
    # Directional Indicators
    plus_di = np.full(len(close_1d), np.nan)
    minus_di = np.full(len(close_1d), np.nan)
    dx = np.full(len(close_1d), np.nan)
    
    for i in range(14, len(close_1d)):
        if not (np.isnan(atr_1d[i]) or atr_1d[i] == 0):
            plus_di[i] = (plus_dm_smooth[i] / atr_1d[i]) * 100
            minus_di[i] = (minus_dm_smooth[i] / atr_1d[i]) * 100
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
    
    # ADX (smoothed DX)
    adx_smooth = np.full(len(close_1d), np.nan)
    # Initial ADX value (first 14 DX values after period 14)
    for i in range(28, len(close_1d)):
        if not np.isnan(dx[i-13:i+1]).any():
            adx_smooth[i] = np.mean(dx[i-13:i+1])
            break
    
    # Wilder's smoothing for subsequent ADX values
    for i in range(29, len(close_1d)):
        if not (np.isnan(adx_smooth[i-1]) or np.isnan(dx[i])):
            adx_smooth[i] = (adx_smooth[i-1] * 13 + dx[i]) / 14
    
    adx = adx_smooth
    
    # Align all HTF/LTF indicators to 6h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, prices, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, prices, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, prices, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, prices, camarilla_l4)
    camarilla_h3_ub_aligned = align_htf_to_ltf(prices, prices, camarilla_h3_ub)
    camarilla_l3_lb_aligned = align_htf_to_ltf(prices, prices, camarilla_l3_lb)
    volume_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_type = 0  # 1=breakout, 2=mean_reversion, 0=none
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(volume_ma_20_12h_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period MA
        volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        volume_confirm = volume_12h_aligned[i] > 1.5 * volume_ma_20_12h_aligned[i]
        
        # Regime filters
        adx_trending = adx_aligned[i] > 25   # Trending market (use breakout)
        adx_ranging = adx_aligned[i] < 20    # Ranging market (use mean reversion)
        
        if position == 0:  # Flat - look for new entries
            # Breakout entries (trending market)
            if adx_trending and volume_confirm:
                # Long breakout: Price breaks above H4
                if close[i] > camarilla_h4_aligned[i]:
                    position = 1
                    entry_type = 1
                    signals[i] = 0.25
                # Short breakout: Price breaks below L4
                elif close[i] < camarilla_l4_aligned[i]:
                    position = -1
                    entry_type = 1
                    signals[i] = -0.25
            # Mean reversion entries (ranging market)
            elif adx_ranging:
                # Long mean reversion: Price below L3 (oversold)
                if close[i] < camarilla_l3_aligned[i]:
                    position = 1
                    entry_type = 2
                    signals[i] = 0.25
                # Short mean reversion: Price above H3 (overbought)
                elif close[i] > camarilla_h3_aligned[i]:
                    position = -1
                    entry_type = 2
                    signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            if entry_type == 1:  # Breakout position
                # Exit breakout when price returns to midpoint of the range
                midpoint = (camarilla_h4_aligned[i] + camarilla_l4_aligned[i]) / 2
                if position == 1:  # Long breakout
                    if close[i] <= midpoint:
                        position = 0
                        entry_type = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = 0.25
                else:  # Short breakout
                    if close[i] >= midpoint:
                        position = 0
                        entry_type = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = -0.25
            else:  # entry_type == 2 (Mean reversion position)
                # Exit mean reversion when price reaches the opposite level
                if position == 1:  # Long mean reversion (from L3)
                    if close[i] >= camarilla_h3_ub_aligned[i]:  # Reached H3
                        position = 0
                        entry_type = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = 0.25
                else:  # Short mean reversion (from H3)
                    if close[i] <= camarilla_l3_lb_aligned[i]:  # Reached L3
                        position = 0
                        entry_type = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = -0.25
    
    return signals