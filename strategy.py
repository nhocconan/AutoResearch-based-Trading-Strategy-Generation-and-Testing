#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation.
# Uses Ichimoku (Tenkan/Kijun cross + price vs cloud) for trend/follow signals,
# filtered by 1d EMA50 trend direction and volume spikes to avoid false signals.
# Works in bull/bear markets by following 1d trend while using Ichimoku as structure.
# Target: 50-150 total trades over 4 years (12-37/year). Size: 0.25.

name = "6h_Ichimoku_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid TypeError
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h Ichimoku components
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
    
    # Align Senkou spans (they are plotted 26 periods ahead, so shift back for current cloud)
    # For current cloud, we use Senkou A/B calculated 26 periods ago
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    # 6h volume spike: >1.5x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 78  # Ichimoku needs max(52,26)+26 for lagged spans
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(tenkan[i]) or
            np.isnan(kijun[i]) or
            np.isnan(senkou_a_lagged[i]) or
            np.isnan(senkou_b_lagged[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d EMA50 direction
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Ichimoku signals
        # Bullish: Tenkan > Kijun AND price > Cloud (above Senkou Span A and B)
        bullish_cross = tenkan[i] > kijun[i]
        bullish_price_vs_cloud = close[i] > max(senkou_a_lagged[i], senkou_b_lagged[i])
        
        # Bearish: Tenkan < Kijun AND price < Cloud (below Senkou Span A and B)
        bearish_cross = tenkan[i] < kijun[i]
        bearish_price_vs_cloud = close[i] < min(senkou_a_lagged[i], senkou_b_lagged[i])
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        long_entry = bullish_cross and bullish_price_vs_cloud and vol_confirm
        short_entry = bearish_cross and bearish_price_vs_cloud and vol_confirm
        
        # Exit conditions: opposite cross or price re-enters cloud
        long_exit = (tenkan[i] < kijun[i]) or (close[i] < min(senkou_a_lagged[i], senkou_b_lagged[i]))
        short_exit = (tenkan[i] > kijun[i]) or (close[i] > max(senkou_a_lagged[i], senkou_b_lagged[i]))
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals