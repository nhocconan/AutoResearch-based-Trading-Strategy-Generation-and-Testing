#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ATR Regime Filter
# - Primary: 6h Elder Ray (Bull Power = Close - EMA13, Bear Power = EMA13 - Close)
# - HTF regime: 1d ATR(14) / ATR(50) > 1.2 = high volatility (trend mode), < 0.8 = low volatility (range mode)
# - In high volatility: trend follow (long when Bull Power > 0 and rising, short when Bear Power > 0 and rising)
# - In low volatility: mean revert (long when Bull Power < -0.5*ATR6h and turning up, short when Bear Power < -0.5*ATR6h and turning down)
# - Volume confirmation: 6h volume > 1.5x 20-period MA
# - Position sizing: 0.25
# - Works in bull/bear: adapts to volatility regime, uses Elder Ray for momentum/mean reversion, volume confirms participation

name = "6h_1d_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR for 6h (primary)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # first bar
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = close - ema_13  # Close - EMA13
    bear_power = ema_13 - close  # EMA13 - Close
    
    # Calculate 1d ATR regime filter
    tr1d = high_1d - low_1d
    tr1d_hl = np.abs(high_1d - np.roll(close_1d, 1))
    tr1d_ll = np.abs(low_1d - np.roll(close_1d, 1))
    tr1d[0] = high_1d[0] - low_1d[0]
    tr1d_hl[0] = np.abs(high_1d[0] - close_1d[0])
    tr1d_ll[0] = np.abs(low_1d[0] - close_1d[0])
    tr_1d = np.maximum(tr1d, np.maximum(tr1d_hl, tr1d_ll))
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_50_1d = pd.Series(tr_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio_1d = atr_14_1d / np.where(atr_50_1d != 0, atr_50_1d, 1e-10)
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Calculate 6h volume MA(20)
    volume_ma_20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(60, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(atr_6h[i]) or np.isnan(atr_ratio_1d_aligned[i]) or
            np.isnan(volume_ma_20_6h[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-period MA
        volume_confirm = volume[i] > 1.5 * volume_ma_20_6h[i]
        
        # Regime filters
        high_vol = atr_ratio_1d_aligned[i] > 1.2  # ATR(14)/ATR(50) > 1.2
        low_vol = atr_ratio_1d_aligned[i] < 0.8   # ATR(14)/ATR(50) < 0.8
        
        if position == 0:  # Flat - look for new entries
            if high_vol:  # High volatility: trend follow
                # Long: Bull Power > 0 and rising (current > previous)
                # Short: Bear Power > 0 and rising (current > previous)
                if bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and volume_confirm:
                    position = 1
                    signals[i] = 0.25
                elif bear_power[i] > 0 and bear_power[i] > bear_power[i-1] and volume_confirm:
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
            elif low_vol:  # Low volatility: mean revert
                # Long: Bull Power < -0.5*ATR6h and turning up (current > previous)
                # Short: Bear Power < -0.5*ATR6h and turning down (current < previous)
                if bull_power[i] < -0.5 * atr_6h[i] and bull_power[i] > bull_power[i-1] and volume_confirm:
                    position = 1
                    signals[i] = 0.25
                elif bear_power[i] < -0.5 * atr_6h[i] and bear_power[i] < bear_power[i-1] and volume_confirm:
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
            else:  # Neutral regime - no trade
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: regime change or power reversal
            if position == 1:  # Long position
                exit_condition = False
                if high_vol and (bull_power[i] <= 0 or bull_power[i] < bull_power[i-1]):
                    exit_condition = True  # Trend follow exit: power <= 0 or falling
                elif low_vol and bull_power[i] >= -0.2 * atr_6h[i]:
                    exit_condition = True  # Mean revert exit: power back to neutral
                elif not (high_vol or low_vol):  # Neutral regime exit
                    exit_condition = True
                
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = False
                if high_vol and (bear_power[i] <= 0 or bear_power[i] < bear_power[i-1]):
                    exit_condition = True  # Trend follow exit: power <= 0 or falling
                elif low_vol and bear_power[i] >= -0.2 * atr_6h[i]:
                    exit_condition = True  # Mean revert exit: power back to neutral
                elif not (high_vol or low_vol):  # Neutral regime exit
                    exit_condition = True
                
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals