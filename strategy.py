#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR-based regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14) for volatility regime
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate daily ATR percentile rank (252-day lookback ~ 1 year)
    # High ATR = volatile regime (trend following), Low ATR = ranging regime (mean reversion)
    atr_percentile = pd.Series(atr_1d).rolling(window=252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    
    # Regime: ATR percentile > 0.6 = volatile/trending regime, < 0.4 = ranging regime
    # We'll use trend following in volatile regimes, mean reversion in ranging regimes
    
    # Get 4-hour data for Donchian channels (our primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian channel (20-period) on 4H data
    highest_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4H Donchian levels to 15-minute timeframe
    highest_high_aligned = align_htf_to_ltf(prices, df_4h, highest_high_4h)
    lowest_low_aligned = align_htf_to_ltf(prices, df_4h, lowest_low_4h)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(50, 252)  # Need sufficient data for ATR percentile
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high_aligned[i]) or 
            np.isnan(lowest_low_aligned[i]) or 
            np.isnan(volume_ma20[i]) or 
            np.isnan(atr_percentile[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Regime filter based on daily ATR percentile
        volatile_regime = atr_percentile[i] > 0.6
        ranging_regime = atr_percentile[i] < 0.4
        
        if position == 0:
            # In volatile regime: trend following - breakout strategy
            if volatile_regime and volume_filter:
                # Long: price breaks above 4H Donchian high
                if close[i] > highest_high_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below 4H Donchian low
                elif close[i] < lowest_low_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            # In ranging regime: mean reversion - fade extreme moves
            elif ranging_regime and volume_filter:
                # Long: price touches or goes below 4H Donchian low (oversold)
                if close[i] <= lowest_low_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price touches or goes above 4H Donchian high (overbought)
                elif close[i] >= highest_high_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit conditions
            if volatile_regime:
                # In volatile regime: exit when price returns to midpoint
                midpoint = (highest_high_aligned[i] + lowest_low_aligned[i]) / 2
                if close[i] <= midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # In ranging regime: exit when price reaches opposite extreme
                if close[i] >= highest_high_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions
            if volatile_regime:
                # In volatile regime: exit when price returns to midpoint
                midpoint = (highest_high_aligned[i] + lowest_low_aligned[i]) / 2
                if close[i] >= midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # In ranging regime: exit when price reaches opposite extreme
                if close[i] <= lowest_low_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "15m_ATRRegime_DonchianBreakout_MeanRev"
timeframe = "15m"
leverage = 1.0