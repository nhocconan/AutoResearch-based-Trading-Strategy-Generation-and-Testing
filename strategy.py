#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Volume + Regime Filter
# - Primary: 6h timeframe for balanced trade frequency
# - HTF: 12h for trend direction (EMA50), 1d for volatility regime (ATR percentile)
# - Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# - Long: Bull Power > 0 + Bear Power increasing (less negative) + 12h Uptrend + Volume Spike
# - Short: Bear Power < 0 + Bull Power decreasing (less positive) + 12h Downtrend + Volume Spike
# - Exit: Elder Ray divergence or opposite signal
# - Position sizing: 0.25 (discrete level)
# - Target: 80-120 total trades over 4 years (20-30/year) - within 6h sweet spot
# - Elder Ray measures bull/bear power relative to EMA, effective in both trending and ranging markets
# - Volume confirmation increases breakout reliability
# - 12h EMA50 ensures we trade with intermediate-term trend
# - 1d ATR percentile filter avoids low-volatility whipsaws

name = "6h_12h_1d_elderray_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 60 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 12h data
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 6h EMA(13) for Elder Ray
    ema_13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high_6h - ema_13_6h  # Bull Power = High - EMA
    bear_power = low_6h - ema_13_6h   # Bear Power = Low - EMA
    
    # Calculate 12h EMA(50) for trend direction
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR percentile rank (using 50-day lookback)
    atr_percentile = pd.Series(atr_1d).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # Calculate 6h volume moving average (20-period) for volume confirmation
    volume_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(volume_ma_20_6h[i])):
            signals[i] = 0.0
            continue
        
        # Elder Ray conditions
        bull_power_pos = bull_power[i] > 0
        bear_power_neg = bear_power[i] < 0
        bull_power_rising = bull_power[i] > bull_power[i-1]  # Increasing bull power
        bear_power_falling = bear_power[i] < bear_power[i-1]  # Decreasing bear power (more negative)
        
        # 12h trend conditions
        uptrend_12h = close_6h[i] > ema_50_12h_aligned[i]
        downtrend_12h = close_6h[i] < ema_50_12h_aligned[i]
        
        # Volatility regime: ATR > 30th percentile (avoid low-vol chop)
        vol_regime = atr_percentile_aligned[i] > 30
        
        # Volume confirmation: current volume > 1.5x 20-period MA
        volume_spike = volume_6h[i] > 1.5 * volume_ma_20_6h[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power positive AND rising + 12h uptrend + vol regime + volume spike
            if (bull_power_pos and bull_power_rising and uptrend_12h and vol_regime and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Bear Power negative AND falling + 12h downtrend + vol regime + volume spike
            elif (bear_power_neg and bear_power_falling and downtrend_12h and vol_regime and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Elder Ray divergence (power weakening)
            # 2. Opposite Elder Ray signal
            # 3. 12h trend changes
            
            if position == 1:  # Long position
                # Exit if bull power weakening or bear power strengthening
                bull_power_falling = bull_power[i] < bull_power[i-1]
                bear_power_rising = bear_power[i] > bear_power[i-1]
                exit_condition = (
                    bull_power_falling or     # Bull power weakening
                    bear_power_rising or      # Bear power strengthening
                    not uptrend_12h           # 12h trend turned down
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                # Exit if bear power weakening or bull power strengthening
                bear_power_rising = bear_power[i] > bear_power[i-1]
                bull_power_falling = bull_power[i] < bull_power[i-1]
                exit_condition = (
                    bear_power_rising or      # Bear power weakening
                    bull_power_falling or     # Bull power strengthening
                    not downtrend_12h         # 12h trend turned up
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals