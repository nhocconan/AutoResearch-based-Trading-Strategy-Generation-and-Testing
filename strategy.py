#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA trend + volume confirmation
# - Primary: 1d timeframe for lower trade frequency (target: 30-100 trades over 4 years)
# - HTF: 1w for trend filter (EMA34)
# - Long: Price > Donchian upper(20) + close > 1w EMA34 + volume > 1.5x 20-day MA
# - Short: Price < Donchian lower(20) + close < 1w EMA34 + volume > 1.5x 20-day MA
# - Exit: Price crosses Donchian midpoint (10-day avg of upper/lower) or ATR < 30th percentile
# - Position sizing: 0.25 (discrete level)
# - Works in bull/bear: Donchian captures breakouts, 1w EMA filters counter-trend, volume/ATR regimes avoid chop

name = "1d_1w_donchian_ema_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
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
    
    # Calculate 1d Donchian channels (20-period)
    # Upper: 20-period high, Lower: 20-period low
    donch_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_upper + donch_lower) / 2.0  # Midpoint for exit
    
    # Calculate 1w EMA(34) for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR percentile rank (using 30-bar lookback)
    atr_percentile = pd.Series(atr_1d).rolling(window=30, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr_percentile[i]) or 
            np.isnan(volume_ma_20_1d[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # Volatility regime: ATR > 30th percentile (avoid extremely low vol)
        vol_regime = atr_percentile[i] > 30
        
        # Volume confirmation: current volume > 1.5x 20-period MA
        volume_spike = volume_1d[i] > 1.5 * volume_ma_20_1d[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price > Donchian upper + close > 1w EMA + vol regime + volume spike
            if (close_1d[i] > donch_upper[i] and 
                close_1d[i] > ema_34_1w_aligned[i] and 
                vol_regime and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Price < Donchian lower + close < 1w EMA + vol regime + volume spike
            elif (close_1d[i] < donch_lower[i] and 
                  close_1d[i] < ema_34_1w_aligned[i] and 
                  vol_regime and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price crosses Donchian midpoint (trend weakening)
            # 2. ATR falls below 30th percentile (low volatility regime)
            
            if position == 1:  # Long position
                exit_condition = (
                    close_1d[i] < donch_mid[i] or  # Price crossed below midpoint
                    atr_percentile[i] < 30  # Low volatility regime
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    close_1d[i] > donch_mid[i] or  # Price crossed above midpoint
                    atr_percentile[i] < 30  # Low volatility regime
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals