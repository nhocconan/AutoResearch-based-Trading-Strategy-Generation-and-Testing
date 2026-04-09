#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1w ATR-based volatility regime filter and 1d Donchian channel breakouts
# - Uses 1w ATR(20) normalized by price to determine volatility regime (high/low vol)
# - Uses 1d Donchian(20) for breakout signals in direction of 1w trend (EMA50)
# - In high volatility regime ( ATR/price > 0.03 ): trade breakouts with 1w trend filter
# - In low volatility regime ( ATR/price <= 0.03 ): fade Donchian touches (mean reversion)
# - Volume confirmation: current 6h volume > 1.5x 20-period average to avoid low-volume false signals
# - Fixed position size 0.25 to control drawdown
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets via regime adaptation

name = "6h_1d_1w_atr_regime_donchian_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ATR(20) for volatility regime
    # True Range = max(high-low, abs(high-previous_close), abs(low-previous_close))
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = tr2[0] = tr3[0] = 0  # first period has no previous close
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(true_range).rolling(window=20, min_periods=20).mean().values
    
    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # 1d Donchian Channel (20 periods)
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_mid_20 = (donch_high_20 + donch_low_20) / 2
    
    # Align all HTF data to 6h timeframe (wait for completed HTF bar)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    donch_mid_20_aligned = align_htf_to_ltf(prices, df_1d, donch_mid_20)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_1w_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or
            np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or
            np.isnan(donch_mid_20_aligned[i]) or np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0 or close[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Volatility regime: ATR/price ratio
        atr_ratio = atr_1w_aligned[i] / close[i]
        high_vol_regime = atr_ratio > 0.03  # High volatility regime
        
        # 1w trend filter: price above/below EMA50
        bullish_trend_1w = close[i] > ema50_1w_aligned[i]
        bearish_trend_1w = close[i] < ema50_1w_aligned[i]
        
        # Donchian levels
        upper_donch = donch_high_20_aligned[i]
        lower_donch = donch_low_20_aligned[i]
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit conditions
            if high_vol_regime:
                # High vol: exit on Donchian middle touch or trend change
                if close[i] <= donch_mid_20_aligned[i] or not bullish_trend_1w:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            else:
                # Low vol: exit on Donchian upper touch or trend change
                if close[i] >= upper_donch or not bullish_trend_1w:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
                    
        elif position == -1:  # Short position
            # Exit conditions
            if high_vol_regime:
                # High vol: exit on Donchian middle touch or trend change
                if close[i] >= donch_mid_20_aligned[i] or not bearish_trend_1w:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
            else:
                # Low vol: exit on Donchian lower touch or trend change
                if close[i] <= lower_donch or not bearish_trend_1w:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
        else:  # Flat
            # Entry logic based on regime and Donchian breakout/fade
            if volume_confirmed:
                if high_vol_regime:
                    # High volatility regime: trade breakouts with 1w trend
                    if bullish_trend_1w and close[i] > upper_donch:
                        position = 1
                        signals[i] = position_size
                    elif bearish_trend_1w and close[i] < lower_donch:
                        position = -1
                        signals[i] = -position_size
                else:
                    # Low volatility regime: fade Donchian touches (mean reversion)
                    if close[i] <= lower_donch and bullish_trend_1w:
                        # Near lower Donchian in bullish 1w trend: long mean reversion
                        position = 1
                        signals[i] = position_size
                    elif close[i] >= upper_donch and bearish_trend_1w:
                        # Near upper Donchian in bearish 1w trend: short mean reversion
                        position = -1
                        signals[i] = -position_size
    
    return signals