#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ATR filter and volume confirmation.
# Long when price breaks above 20-period high AND ATR(14) < 0.5 * ATR(50) (low volatility regime) AND volume > 1.5x 20-period average.
# Short when price breaks below 20-period low AND ATR(14) < 0.5 * ATR(50) AND volume > 1.5x 20-period average.
# Uses discrete position size 0.25. Donchian breakouts capture momentum, ATR filter avoids high volatility whipsaws, volume spike confirms participation.
# Designed for 12h timeframe to minimize fee drag while capturing significant moves in both bull and bear markets.
# Target: 80-120 trades over 4 years (20-30/year) to balance opportunity and cost.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Donchian Channel (20) ===
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_ma.values
    donchian_low = low_ma.values
    
    # === 12h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # === 12h Indicators: ATR(14) and ATR(50) for volatility filter ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    atr_50 = pd.Series(tr).ewm(alpha=1/50, adjust=False, min_periods=50).mean()
    atr_ratio = (atr_14 / atr_50).values  # Low volatility when ratio < 0.5
    
    # Get 1d data once before loop for additional confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: EMA(50) for trend filter ===
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA, 20 for Donchian/volume, 50 for ATR)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr_ratio[i]) or np.isnan(vol_ma[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        vol_spike = volume_spike[i]
        low_vol_regime = atr_ratio[i] < 0.5
        ema_50 = ema_50_1d_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Donchian low or volatility increases
            if price < lower_band or not low_vol_regime:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Donchian high or volatility increases
            if price > upper_band or not low_vol_regime:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian high AND low volatility regime AND volume spike AND price > 1d EMA50 (uptrend)
            if price > upper_band and low_vol_regime and vol_spike and price > ema_50:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Donchian low AND low volatility regime AND volume spike AND price < 1d EMA50 (downtrend)
            elif price < lower_band and low_vol_regime and vol_spike and price < ema_50:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_Donchian20_1dATR_Volume_EMA50_V1"
timeframe = "12h"
leverage = 1.0