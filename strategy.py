#!/usr/bin/env python3
"""
1h_Ichimoku_Kumo_Twist_4hTrend_v1
Hypothesis: 1h Ichimoku Kumo twist (Tenkan/Kijun cross) with 4h trend filter and volume confirmation.
- Uses 1h timeframe targeting 60-150 total trades over 4 years (15-37/year)
- Long when Tenkan crosses above Kijun AND price is above cloud AND 4h uptrend AND volume spike
- Short when Tenkan crosses below Kijun AND price is below cloud AND 4h downtrend AND volume spike
- Ichimoku cloud (Senkou Span A/B) acts as dynamic support/resistance
- 4h EMA50 trend filter reduces whipsaw in bear markets and captures major moves
- Volume spike (2.0x 20-period average) confirms institutional participation
- Session filter (08-20 UTC) reduces noise trades outside active market hours
- Designed for medium frequency with proven edge on BTC/ETH from Ichimoku's trend-following nature
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Current cloud boundaries (Senkou Span A/B plotted 26 periods ahead)
    # For signal at index i, we use Senkou values from i-26 (already plotted)
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    # First 26 values are invalid due to lag
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    # Kumo twist: Tenkan/Kijun cross
    tenkan_prev = np.roll(tenkan, 1)
    kijun_prev = np.roll(kijun, 1)
    tenkan_prev[0] = np.nan
    kijun_prev[0] = np.nan
    
    tk_cross_up = (tenkan > kijun) & (tenkan_prev <= kijun_prev)
    tk_cross_down = (tenkan < kijun) & (tenkan_prev >= kijun_prev)
    
    # Price above/below cloud
    price_above_cloud = (close > senkou_a_lagged) & (close > senkou_b_lagged)
    price_below_cloud = (close < senkou_a_lagged) & (close < senkou_b_lagged)
    
    # Calculate 4h EMA50 for trend filter
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate volume spike (20-period volume average on 1h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)  # Volume at least 2.0x average
    
    # Session filter: 08-20 UTC (active market hours)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou B, 20 for volume MA)
    start_idx = max(52, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(tk_cross_up[i]) or np.isnan(tk_cross_down[i]) or
            np.isnan(price_above_cloud[i]) or np.isnan(price_below_cloud[i]) or
            np.isnan(senkou_a_lagged[i]) or np.isnan(senkou_b_lagged[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Ichimoku Kumo twist conditions with volume confirmation, trend filter, and session filter
        if position == 0:
            # Long: Tenkan/Kijun cross up AND price above cloud AND 4h uptrend AND volume spike AND in session
            if tk_cross_up[i] and price_above_cloud[i] and close[i] > ema50_4h_aligned[i] and volume_spike[i] and in_session[i]:
                signals[i] = 0.20
                position = 1
            # Short: Tenkan/Kijun cross down AND price below cloud AND 4h downtrend AND volume spike AND in session
            elif tk_cross_down[i] and price_below_cloud[i] and close[i] < ema50_4h_aligned[i] and volume_spike[i] and in_session[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: Tenkan/Kijun cross down OR price falls below cloud OR 4h trend turns down
            if tk_cross_down[i] or not price_above_cloud[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: Tenkan/Kijun cross up OR price rises above cloud OR 4h trend turns up
            if tk_cross_up[i] or not price_below_cloud[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Ichimoku_Kumo_Twist_4hTrend_v1"
timeframe = "1h"
leverage = 1.0