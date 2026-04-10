#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX regime filter and volume confirmation
# - Long when price breaks above Donchian upper (20) AND 1d ADX > 25 (trending) AND 4h volume > 1.5x 20-bar avg
# - Short when price breaks below Donchian lower (20) AND 1d ADX > 25 (trending) AND 4h volume > 1.5x 20-bar avg
# - Exit when price returns to Donchian midpoint (mean reversion to equilibrium)
# - Uses discrete position sizing (0.25) to balance return and drawdown
# - Donchian channels provide structural support/resistance based on recent price action
# - 1d ADX filter ensures we only trade in trending markets to avoid whipsaws in ranges
# - Volume confirmation avoids low-liquidity false signals
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)

name = "4h_1d_donchian_breakout_adx_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d < 50):
        return np.zeros(n)
    
    # Pre-compute 1d ADX trend filter: ADX > 25 indicates trending market
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    # Calculate Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    plus_di_1d = 100 * wilders_smoothing(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilders_smoothing(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d, 14)
    adx_trending_1d = adx_1d > 25
    
    # Pre-compute 4h volume confirmation: > 1.5x 20-period average
    volume_4h = prices['volume'].values
    volume_20_avg_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike_4h = volume_4h > (1.5 * volume_20_avg_4h)
    
    # Pre-compute 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    donchian_upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid_4h = (donchian_upper_4h + donchian_lower_4h) / 2
    
    # Align HTF indicators to 4h timeframe
    adx_trending_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_trending_1d)
    
    # Session filter: 08-20 UTC (avoid low liquidity Asian session)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(adx_trending_1d_aligned[i]) or np.isnan(donchian_upper_4h[i]) or
            np.isnan(donchian_lower_4h[i]) or np.isnan(donchian_mid_4h[i]) or
            np.isnan(vol_spike_4h[i])):
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
            # Long when price breaks above upper DONCH AND 1d trending AND volume spike
            if (prices['close'].iloc[i] > donchian_upper_4h[i] and 
                adx_trending_1d_aligned[i] and 
                vol_spike_4h[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below lower DONCH AND 1d trending AND volume spike
            elif (prices['close'].iloc[i] < donchian_lower_4h[i] and 
                  adx_trending_1d_aligned[i] and 
                  vol_spike_4h[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to midpoint (mean reversion)
            # Exit when price returns to Donchian midpoint
            exit_long = position == 1 and prices['close'].iloc[i] <= donchian_mid_4h[i]
            exit_short = position == -1 and prices['close'].iloc[i] >= donchian_mid_4h[i]
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals