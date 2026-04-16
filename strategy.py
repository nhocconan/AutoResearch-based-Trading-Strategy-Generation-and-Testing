#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Width regime filter combined with 1d RSI mean reversion and volume spike confirmation.
# In low volatility regimes (BB Width < 20th percentile), RSI extremes (RSI < 30 or > 70) with volume spike (>1.5x 20-period 1d average) indicate high-probability mean reversion trades.
# Long when RSI < 30, volume spike, and low volatility regime. Short when RSI > 70, volume spike, and low volatility regime.
# Uses discrete position size 0.25. Designed to capture mean reversion in low volatility environments, which occur in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Bollinger Band Width (20,2) ===
    bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # === 1d Indicators: RSI(14) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rs = np.where(avg_loss == 0, 100, rs)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 1d Indicators: BB Width Percentile (20,2) for regime filter ===
    bb_middle_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper_1d = bb_middle_1d + 2 * bb_std_1d
    bb_lower_1d = bb_middle_1d - 2 * bb_std_1d
    bb_width_1d = (bb_upper_1d - bb_lower_1d) / bb_middle_1d
    
    # Calculate 20th percentile of BB Width using expanding window (minimum 50 periods)
    bb_width_percentile = np.full_like(bb_width_1d, np.nan)
    for i in range(50, len(bb_width_1d)):
        bb_width_percentile[i] = np.percentile(bb_width_1d[:i+1], 20)
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    low_volatility_regime = bb_width_1d < bb_width_percentile_aligned
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for percentile)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Calculate 6h ATR for stoploss
    tr1_6h = pd.Series(high).diff()
    tr2_6h = pd.Series(low).diff().abs()
    tr3_6h = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1_6h, tr2_6h, tr3_6h], axis=1).max(axis=1)
    atr_6h_raw = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(bb_width[i]) or np.isnan(rsi_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(low_volatility_regime[i]) or np.isnan(atr_6h_raw[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        is_low_vol = low_volatility_regime[i]
        rsi_val = rsi_aligned[i]
        atr_val = atr_6h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if RSI returns to neutral (40-60) or volatility regime changes
            if rsi_val >= 40 and rsi_val <= 60:
                exit_signal = True
            elif not is_low_vol:  # Exit low volatility regime
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if RSI returns to neutral (40-60) or volatility regime changes
            if rsi_val >= 40 and rsi_val <= 60:
                exit_signal = True
            elif not is_low_vol:  # Exit low volatility regime
                exit_signal = True
            # ATR-based stoploss: 2*ATR above entry
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: RSI < 30, volume spike, and low volatility regime
            if rsi_val < 30 and vol_spike and is_low_vol:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: RSI > 70, volume spike, and low volatility regime
            elif rsi_val > 70 and vol_spike and is_low_vol:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_BBWidthRegime_1dRSI_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0