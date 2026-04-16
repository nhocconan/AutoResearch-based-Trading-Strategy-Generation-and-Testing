#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout with 1d volume confirmation and ATR stoploss
# Uses Bollinger Band width percentile to identify low volatility squeezes (regime filter).
# Breakout triggers when price closes outside BB(20,2) AND 1d volume > 1.5x 20-period average.
# ATR-based stoploss (2.0x) and opposite-band exit for risk management.
# Target: 75-200 total trades over 4 years (19-50/year) to balance statistical significance and fee drag.
# Works in bull markets via long breakouts and in bear markets via short breakdowns during volume expansion.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # === 1d data (HTF for volume regime) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 4h Bollinger Bands (20,2) ===
    bb_middle = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band width percentile (252-period lookback ~ 1 year of 4h data)
    bb_width_pct = pd.Series(bb_width).rolling(window=252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    squeeze_condition = bb_width_pct < 0.2  # Low volatility squeeze (<20th percentile)
    squeeze_aligned = align_htf_to_ltf(prices, df_4h, squeeze_condition)
    
    # === 1d Volume regime filter (expanding volume) ===
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume_1d > (1.5 * vol_ma_20_1d)  # True when volume expanding
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    
    # === 4h ATR (14) for stoploss ===
    tr1 = np.abs(high_4h - low_4h)
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(squeeze_aligned[i]) or 
            np.isnan(vol_regime_aligned[i]) or
            np.isnan(atr_aligned[i]) or
            np.isnan(bb_upper[i]) or
            np.isnan(bb_lower[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        squeeze_ok = squeeze_aligned[i]
        regime_ok = vol_regime_aligned[i]
        atr_val = atr_aligned[i]
        
        # === STOPLOSS LOGIC (ATR-based) ===
        if position == 1:  # Long position
            if price < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            if price > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC (opposite Bollinger Band) ===
        if position == 1:  # Long position
            # Exit when price touches or crosses below middle band
            if price < bb_middle[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price touches or crosses above middle band
            if price > bb_middle[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require both squeeze (low volatility) and volume regime (expanding volume)
            if squeeze_ok and regime_ok:
                # Go long when price closes above upper Bollinger Band
                if close_4h[i] > bb_upper[i] and close_4h[i-1] <= bb_upper[i-1]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Go short when price closes below lower Bollinger Band
                elif close_4h[i] < bb_lower[i] and close_4h[i-1] >= bb_lower[i-1]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_BBSqueeze_Breakout_VolumeRegime_ATRStop"
timeframe = "4h"
leverage = 1.0