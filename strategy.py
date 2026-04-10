#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d ADX trend filter and volume confirmation
# - Long when price breaks above 1d Donchian(20) upper band AND 1d ADX(14) > 25 (strong trend) AND 1d volume > 1.5x 20-bar avg
# - Short when price breaks below 1d Donchian(20) lower band AND 1d ADX(14) > 25 (strong trend) AND 1d volume > 1.5x 20-bar avg
# - Exit when price crosses the 1d Donchian midpoint (mean reversion to equilibrium)
# - Uses discrete position sizing (0.25) to balance return and drawdown
# - Donchian channels provide clear trend structure based on recent highs/lows
# - 1d ADX filter ensures we only trade in strong trending markets to avoid chop
# - Volume confirmation adds conviction to breakouts
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets: breakouts capture trends, mean reversion exits work in ranging markets

name = "12h_1d_donchian_breakout_adx_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian upper band: highest high of last 20 periods
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian lower band: lowest low of last 20 periods
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    # Donchian midpoint: average of upper and lower bands
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Pre-compute 1d ADX (14-period) for trend strength
    # Calculate True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Calculate Directional Movement
    up_move = pd.Series(high_1d - np.roll(high_1d, 1))
    down_move = pd.Series(np.roll(low_1d, 1) - low_1d)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Calculate DI and DX
    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Strong trend condition: ADX > 25
    strong_trend = adx.values > 25
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * volume_20_avg_1d)
    
    # Align HTF indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    strong_trend_aligned = align_htf_to_ltf(prices, df_1d, strong_trend)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Session filter: 08-20 UTC (avoid low liquidity Asian session)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or np.isnan(strong_trend_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i])):
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
            # Long when price breaks above Donchian high AND strong trend AND volume spike
            if (prices['close'].iloc[i] > donchian_high_aligned[i] and 
                strong_trend_aligned[i] and 
                vol_spike_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian low AND strong trend AND volume spike
            elif (prices['close'].iloc[i] < donchian_low_aligned[i] and 
                  strong_trend_aligned[i] and 
                  vol_spike_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to Donchian midpoint (mean reversion)
            # Exit when price crosses the Donchian midpoint
            exit_long = position == 1 and prices['close'].iloc[i] <= donchian_mid_aligned[i]
            exit_short = position == -1 and prices['close'].iloc[i] >= donchian_mid_aligned[i]
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals