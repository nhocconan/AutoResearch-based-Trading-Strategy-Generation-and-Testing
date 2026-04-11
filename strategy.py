#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d volume spike and 1w ADX trend filter
# - Enter long when Williams %R(14) < -80 (oversold) AND 1d volume > 2.0x 20-period volume SMA AND 1w ADX < 25 (range/chop regime)
# - Enter short when Williams %R(14) > -20 (overbought) AND 1d volume > 2.0x 20-period volume SMA AND 1w ADX < 25
# - Exit: Williams %R crosses back above -50 (for longs) or below -50 (for shorts)
# - Williams %R identifies extreme price levels for mean reversion
# - Volume confirmation ensures institutional participation at extremes
# - 1w ADX < 25 filter avoids trending markets where mean reversion fails
# - Target: 15-25 trades/year to minimize fee drag while capturing high-probability reversals

name = "12h_1d_1w_williamsr_volspike_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for volume confirmation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Load 1w data ONCE before loop for ADX trend filter (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Pre-compute Williams %R for 12h data (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Pre-compute volume SMA for 1d data (20-period)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute ADX for 1w data (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = pd.Series(high_1w).shift(1).subtract(close_1w).abs()
    tr2 = pd.Series(low_1w).shift(1).subtract(close_1w).abs()
    tr3 = (high_1w - low_1w).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # Directional Movement
    dm_plus = pd.Series(high_1w).diff()
    dm_minus = -pd.Series(low_1w).diff()
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align indicators to 12h timeframe (wait for completed 1d/1w bar)
    williams_r_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low, 'close': close}), williams_r)
    volume_1d_current = df_1d['volume'].values
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_current)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    for i in range(30, n):  # Start after 30-bar warmup for ADX
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(volume_1d_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 2.0x 20-period volume SMA
        vol_confirm = volume_1d_aligned[i] > 2.0 * volume_sma_20_1d_aligned[i]
        
        # Trend filter: 1w ADX < 25 (range/chop regime)
        range_regime = adx_aligned[i] < 25
        
        # Williams %R signals
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        exit_long = williams_r_aligned[i] > -50  # Exit long when %R crosses above -50
        exit_short = williams_r_aligned[i] < -50  # Exit short when %R crosses below -50
        
        # Trading logic
        if vol_confirm and range_regime:
            # Long: Oversold in range regime
            if oversold:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Short: Overbought in range regime
            elif overbought:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:
                # Check for exits
                if position == 1 and exit_long:
                    position = 0
                    signals[i] = 0.0
                elif position == -1 and exit_short:
                    position = 0
                    signals[i] = 0.0
                else:
                    # Maintain current position
                    signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            # No volume confirmation or not in range regime: exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals