#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 12h volume confirmation and choppiness regime filter
# - Long when price breaks above Donchian(20) high AND 12h volume > 1.5x 20-bar avg AND choppiness < 61.8 (trending regime)
# - Short when price breaks below Donchian(20) low AND 12h volume > 1.5x 20-bar avg AND choppiness < 61.8
# - Exit when price returns to Donchian midpoint (mean reversion to equilibrium)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Donchian provides clear structure with proven edge in crypto
# - 12h volume confirmation ensures institutional participation
# - Choppiness filter avoids whipsaws in ranging markets
# - Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe

name = "4h_12h_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h volume confirmation: > 1.5x 20-period average
    volume_12h = df_12h['volume'].values
    volume_20_avg_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume_12h > (1.5 * volume_20_avg_12h)
    
    # Pre-compute 12h choppiness regime filter: CHOP < 61.8 = trending (favor breakouts)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range calculation
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14) calculation
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR over last 14 periods
    atr_sum_12h = pd.Series(atr_12h).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over last 14 periods
    hh_12h = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll_12h = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(atr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero
    hh_ll_diff = hh_12h - ll_12h
    chop_12h = np.where(
        (hh_ll_diff > 0) & (~np.isnan(atr_sum_12h)) & (atr_sum_12h > 0),
        100 * np.log10(atr_sum_12h / hh_ll_diff) / np.log10(14),
        100  # Default to ranging when invalid
    )
    
    # Trending regime: CHOP < 61.8
    trending_regime_12h = chop_12h < 61.8
    
    # Align HTF indicators to 4h timeframe
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    trending_regime_12h_aligned = align_htf_to_ltf(prices, df_12h, trending_regime_12h)
    
    # Pre-compute 4h Donchian channels
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    donchian_period = 20
    donchian_high = pd.Series(high_4h).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low_4h).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Session filter: 08-20 UTC (avoid low liquidity Asian session)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(vol_spike_12h_aligned[i]) or np.isnan(trending_regime_12h_aligned[i])):
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
            # Long when price breaks above Donchian high AND 12h volume spike AND trending regime
            if (close_4h[i] > donchian_high[i] and 
                vol_spike_12h_aligned[i] and 
                trending_regime_12h_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian low AND 12h volume spike AND trending regime
            elif (close_4h[i] < donchian_low[i] and 
                  vol_spike_12h_aligned[i] and 
                  trending_regime_12h_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to Donchian midpoint (mean reversion)
            # Exit when price returns to Donchian midpoint
            exit_long = position == 1 and close_4h[i] <= donchian_mid[i]
            exit_short = position == -1 and close_4h[i] >= donchian_mid[i]
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals