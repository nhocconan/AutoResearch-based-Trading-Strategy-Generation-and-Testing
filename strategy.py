#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
# - Long when price breaks above 6h Donchian upper band (20-bar high) AND weekly pivot shows bullish bias (price > weekly VWAP) AND 6h volume > 1.5x 20-bar avg
# - Short when price breaks below 6h Donchian lower band (20-bar low) AND weekly pivot shows bearish bias (price < weekly VWAP) AND 6h volume > 1.5x 20-bar avg
# - Exit when price crosses 6h Donchian midpoint (mean reversion to equilibrium)
# - Uses discrete position sizing (0.25) to balance return and drawdown
# - Donchian provides clear breakout levels based on recent price action
# - Weekly VWAP filter ensures alignment with higher timeframe trend to avoid counter-trend trades
# - Volume confirmation avoids low-liquidity false signals
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets: breakouts in trends, mean reversion in ranges

name = "6h_1w_donchian_breakout_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w VWAP for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Typical price
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    # VWAP calculation
    vwap_numerator = np.cumsum(typical_price_1w * volume_1w)
    vwap_denominator = np.cumsum(volume_1w)
    vwap_1w = np.divide(vwap_numerator, vwap_denominator, 
                        out=np.full_like(vwap_numerator, np.nan), 
                        where=vwap_denominator!=0)
    vwap_bullish_1w = close_1w > vwap_1w
    vwap_bearish_1w = close_1w < vwap_1w
    
    # Pre-compute 6h Donchian channels (20-period)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    donchian_upper = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Pre-compute 6h volume confirmation: > 1.5x 20-period average
    volume_6h = prices['volume'].values
    volume_20_avg_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike_6h = volume_6h > (1.5 * volume_20_avg_6h)
    
    # Align HTF indicators to 6h timeframe
    vwap_bullish_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_bullish_1w)
    vwap_bearish_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_bearish_1w)
    
    # Session filter: 08-20 UTC (avoid low liquidity Asian session)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(vol_spike_6h[i]) or
            np.isnan(vwap_bullish_1w_aligned[i]) or np.isnan(vwap_bearish_1w_aligned[i])):
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
            # Long when price breaks above upper band AND weekly bullish bias AND volume spike
            if (prices['close'].iloc[i] > donchian_upper[i] and 
                vwap_bullish_1w_aligned[i] and 
                vol_spike_6h[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below lower band AND weekly bearish bias AND volume spike
            elif (prices['close'].iloc[i] < donchian_lower[i] and 
                  vwap_bearish_1w_aligned[i] and 
                  vol_spike_6h[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to midpoint (mean reversion)
            # Exit when price crosses Donchian midpoint
            exit_long = position == 1 and prices['close'].iloc[i] <= donchian_mid[i]
            exit_short = position == -1 and prices['close'].iloc[i] >= donchian_mid[i]
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals