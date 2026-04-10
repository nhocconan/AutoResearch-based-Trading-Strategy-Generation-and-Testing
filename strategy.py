#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 12h EMA trend + volume confirmation
# - Primary: 4h timeframe for optimal trade frequency (target: 75-200 trades over 4 years)
# - HTF: 12h for EMA(34) trend filter
# - Long: Price breaks above Donchian(20) upper band + close > 12h EMA(34) + volume > 1.5x 20-period MA
# - Short: Price breaks below Donchian(20) lower band + close < 12h EMA(34) + volume > 1.5x 20-period MA
# - Exit: Price returns to Donchian(20) middle band or ATR(14) < 20th percentile (low volatility)
# - Position sizing: 0.25 (discrete level)
# - Works in bull/bear: Donchian captures breakouts, 12h EMA avoids counter-trend trades, volume confirms momentum

name = "4h_12h_donchian_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    # Pre-compute 4h OHLCV
    open_4h = prices['open'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    # Pre-compute 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate 12h EMA(34) for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 4h ATR(14) for volatility regime filter
    tr1 = pd.Series(high_4h).shift(1) - pd.Series(low_4h).shift(1)
    tr2 = abs(pd.Series(high_4h) - pd.Series(close_4h).shift(1))
    tr3 = abs(pd.Series(low_4h) - pd.Series(close_4h).shift(1))
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h = tr_4h.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h ATR percentile rank (using 30-bar lookback)
    atr_percentile = pd.Series(atr_4h).rolling(window=30, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Calculate 4h volume moving average (20-period) for volume confirmation
    volume_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(atr_percentile[i]) or 
            np.isnan(volume_ma_20_4h[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # Volatility regime: ATR > 20th percentile (avoid extremely low-vol chop)
        vol_regime = atr_percentile[i] > 20
        
        # Volume confirmation: current 4h volume > 1.5x 20-period MA
        volume_spike = volume_4h[i] > 1.5 * volume_ma_20_4h[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian upper + above 12h EMA + vol regime + volume spike
            if (close_4h[i] > donchian_upper[i] and 
                close_4h[i] > ema_34_12h_aligned[i] and 
                vol_regime and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian lower + below 12h EMA + vol regime + volume spike
            elif (close_4h[i] < donchian_lower[i] and 
                  close_4h[i] < ema_34_12h_aligned[i] and 
                  vol_regime and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price returns to Donchian middle band (mean reversion)
            # 2. ATR falls below 20th percentile (low volatility regime)
            
            if position == 1:  # Long position
                exit_condition = (
                    close_4h[i] < donchian_middle[i] or  # Price returned to middle band
                    atr_percentile[i] < 20  # Low volatility regime
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    close_4h[i] > donchian_middle[i] or  # Price returned to middle band
                    atr_percentile[i] < 20  # Low volatility regime
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals