#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Donchian Breakout with Volume Confirmation and ATR Regime Filter
# - Primary: 1d timeframe for lower trade frequency (7-25/year target)
# - HTF: 1w for multi-week trend context and volatility regime
# - Long: Price breaks above weekly Donchian(20) high + 1w ATR > 50th percentile + volume > 1.5x 20-period MA
# - Short: Price breaks below weekly Donchian(20) low + 1w ATR > 50th percentile + volume > 1.5x 20-period MA
# - Exit: Price reverts to weekly midpoint (mean reversion) or breaks opposite Donchian band
# - Position sizing: 0.25 (discrete level)
# - Works in bull/bear: Donchian breakouts capture trends, weekly filter avoids false breakouts in ranging markets

name = "1d_1w_donchian_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d OHLCV
    open_1d = prices['open'].values
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    volume_1d = prices['volume'].values
    
    # Pre-compute 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w Donchian Channels (20-period)
    # Using rolling window on 1w data
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0  # Weekly midpoint for exit
    
    # Align 1w Donchian levels to 1d bars (using previous completed 1w bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # Calculate 1w ATR(14) for volatility regime filter
    tr1 = pd.Series(high_1w).shift(1) - pd.Series(low_1w).shift(1)
    tr2 = abs(pd.Series(high_1w) - pd.Series(close_1w).shift(1))
    tr3 = abs(pd.Series(low_1w) - pd.Series(close_1w).shift(1))
    tr_1w = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = tr_1w.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1w ATR percentile rank (using 30-bar lookback)
    atr_percentile = pd.Series(atr_1w).rolling(window=30, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1w, atr_percentile)
    
    # Calculate 1w volume moving average (20-period) for volume confirmation
    volume_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(volume_ma_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # 1w volatility regime: ATR > 50th percentile (avoid low-vol chop)
        vol_regime = atr_percentile_aligned[i] > 50
        
        # Volume confirmation: current 1w volume > 1.5x 20-period MA
        volume_spike = volume_1w[i // (7*24*4)] > 1.5 * volume_ma_20_1w_aligned[i]  # Approximate 1w index
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above weekly Donchian high + vol regime + volume spike
            if (close_1d[i] > donchian_high_aligned[i] and vol_regime and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below weekly Donchian low + vol regime + volume spike
            elif (close_1d[i] < donchian_low_aligned[i] and vol_regime and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price reverts to weekly midpoint (mean reversion)
            # 2. Price breaks opposite Donchian band (take profit)
            
            if position == 1:  # Long position
                exit_condition = (
                    close_1d[i] < donchian_mid_aligned[i] or  # Reverted to midpoint
                    close_1d[i] > donchian_high_aligned[i] * 1.02  # Break above high with buffer (take profit)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    close_1d[i] > donchian_mid_aligned[i] or  # Reverted to midpoint
                    close_1d[i] < donchian_low_aligned[i] * 0.98  # Break below low with buffer (take profit)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals