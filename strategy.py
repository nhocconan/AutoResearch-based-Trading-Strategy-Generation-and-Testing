#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and choppiness regime filter
# - Long when price breaks above Donchian(20) high AND 1d volume > 1.5x 20-bar avg AND chop < 61.8 (trending)
# - Short when price breaks below Donchian(20) low AND 1d volume > 1.5x 20-bar avg AND chop < 61.8 (trending)
# - Exit when price crosses Donchian midpoint OR chop > 61.8 (range) OR volume dries up
# - Uses ATR-based stoploss (signal → 0 when price moves against position by 2.0*ATR)
# - Target: 75-200 total trades over 4 years (19-50/year) for BTC/ETH/SOL
# - Works in both bull and bear markets: Donchian captures breakouts, volume confirms strength, chop filter avoids false signals in ranges

name = "4h_1d_donchian_volume_chop_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * volume_20_avg_1d)
    
    # Pre-compute 1d choppiness index: CHOP(14) = 100 * LOG10(SUM(ATR(1),14) / (LOG10(HIGHest HIGH - LOWest LOW,14))) / LOG10(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for ATR
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1 = pd.Series(tr).rolling(window=1, min_periods=1).mean().values  # ATR(1) = TR
    
    # Sum of ATR(1) over 14 periods
    sum_atr_14 = pd.Series(atr_1).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness index
    chop_denominator = highest_high_14 - lowest_low_14
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)  # Avoid division by zero
    chop_raw = 100 * np.log10(sum_atr_14 / chop_denominator) / np.log10(14)
    chop = chop_raw  # Already in 0-100 range
    
    # Regime filter: chop < 61.8 = trending (favor breakouts), chop > 61.8 = ranging (avoid breakouts)
    chop_trending = chop < 61.8
    
    # Align HTF indicators to 4h timeframe
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    chop_trending_aligned = align_htf_to_ltf(prices, df_1d, chop_trending)
    
    # Pre-compute 4h Donchian channels: DC(20) = 20-period high/low
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Pre-compute ATR(10) for stoploss
    tr_4h1 = high_4h[1:] - low_4h[1:]
    tr_4h2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr_4h3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.concatenate([[np.max([high_4h[0] - low_4h[0], np.abs(high_4h[0] - close_4h[0]), np.abs(low_4h[0] - close_4h[0])])], np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))])
    atr_10 = pd.Series(tr_4h).rolling(window=10, min_periods=10).mean().values
    
    # Session filter: 08-20 UTC (reduce noise outside active hours)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0  # Track entry price for stoploss
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(vol_spike_1d_aligned[i]) or np.isnan(chop_trending_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(atr_10[i])):
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
            entry_price = 0.0
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian high AND volume spike AND trending regime
            if (close_4h[i] > donchian_high[i-1] and  # Use previous bar's Donchian high for breakout
                vol_spike_1d_aligned[i] and 
                chop_trending_aligned[i]):
                position = 1
                entry_price = close_4h[i]
                signals[i] = 0.25
            # Short when price breaks below Donchian low AND volume spike AND trending regime
            elif (close_4h[i] < donchian_low[i-1] and  # Use previous bar's Donchian low for breakout
                  vol_spike_1d_aligned[i] and 
                  chop_trending_aligned[i]):
                position = -1
                entry_price = close_4h[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - manage exit
            # Exit conditions:
            # 1. Price crosses Donchian midpoint (mean reversion)
            # 2. Chop regime shifts to ranging (chop > 61.8)
            # 3. Stoploss: price moves against position by 2.0*ATR
            exit_signal = False
            
            if position == 1:  # Long position
                # Exit long if price crosses below midpoint
                if close_4h[i] < donchian_mid[i]:
                    exit_signal = True
                # Exit long if chop shifts to ranging
                elif not chop_trending_aligned[i]:
                    exit_signal = True
                # Stoploss: price drops below entry - 2.0*ATR
                elif close_4h[i] < entry_price - 2.0 * atr_10[i]:
                    exit_signal = True
            else:  # Short position
                # Exit short if price crosses above midpoint
                if close_4h[i] > donchian_mid[i]:
                    exit_signal = True
                # Exit short if chop shifts to ranging
                elif not chop_trending_aligned[i]:
                    exit_signal = True
                # Stoploss: price rises above entry + 2.0*ATR
                elif close_4h[i] > entry_price + 2.0 * atr_10[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals