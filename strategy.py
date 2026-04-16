#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout with 1d volume confirmation and ATR trailing stop.
# Long when price closes above upper BB AND BB width < 20th percentile (squeeze) AND 1d volume > 1.5x 20-period average.
# Short when price closes below lower BB AND BB width < 20th percentile AND 1d volume > 1.5x 20-period average.
# Exit on ATR trailing stop (3*ATR from extreme) or opposite BB touch.
# Uses discrete position size 0.25. Squeeze filter reduces false breakouts in ranging markets.
# Volume confirmation ensures institutional participation. Works in both bull and bear markets.
# Target: 75-200 total trades over 4 years (19-50/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Bollinger Bands (20, 2.0) ===
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean().values
    dev = close_s.rolling(window=20, min_periods=20).std().values
    upper_band = basis + 2.0 * dev
    lower_band = basis - 2.0 * dev
    bb_width = (upper_band - lower_band) / basis  # normalized width
    
    # BB width percentile (20-period lookback) for squeeze detection
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    squeeze_condition = bb_width_percentile < 0.20  # BB width in lowest 20%
    
    # === 1d Indicators: Volume Spike ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 4h ATR for trailing stop ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h_raw = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed)
    warmup = 50
    
    # Track position state, entry price, and extreme for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    extreme_price = 0.0  # highest high for long, lowest low for short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(squeeze_condition[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr_4h_raw[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_4h_raw[i]
        squeeze = squeeze_condition[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price touches lower BB (mean reversion in squeeze)
            if price < lower_band[i]:
                exit_signal = True
            # ATR trailing stop: 3*ATR below extreme
            elif price < extreme_price - 3.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price touches upper BB (mean reversion in squeeze)
            if price > upper_band[i]:
                exit_signal = True
            # ATR trailing stop: 3*ATR above extreme
            elif price > extreme_price + 3.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            extreme_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price closes above upper BB AND squeeze AND volume spike
            if close[i] > upper_band[i] and squeeze and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
                extreme_price = price
            
            # SHORT: Price closes below lower BB AND squeeze AND volume spike
            elif close[i] < lower_band[i] and squeeze and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
                extreme_price = price
        
        else:
            # Update extreme for trailing stop
            if position == 1:
                extreme_price = max(extreme_price, high[i])
            else:  # position == -1
                extreme_price = min(extreme_price, low[i])
            signals[i] = position * 0.25
    
    return signals

name = "4h_BollingerSqueeze_1dVolumeSpike_ATRTrail_V1"
timeframe = "4h"
leverage = 1.0