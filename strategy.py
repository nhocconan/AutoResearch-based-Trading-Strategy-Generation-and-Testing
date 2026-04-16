#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout with volume confirmation and ATR stoploss
# Uses 4h primary timeframe with 12h HTF for Bollinger Band width percentile (squeeze detection).
# Bollinger Band Width (BBW) percentile < 20% indicates low volatility squeeze.
# Breakout: price closes outside BB(20,2) with volume > 1.5x 20-period average.
# Direction: long if price > upper band, short if price < lower band.
# ATR-based stoploss (2.0x) and time-based exit (max 8 bars) for risk management.
# Target: 80-150 total trades over 4 years (20-38/year) to balance edge capture and fee drag.
# Works in bull markets via upside breakouts and in bear markets via downside breakouts during expansion phases.

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
    
    # === 12h data (HTF for BBW percentile) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # === 4h Bollinger Bands (20,2) ===
    bb_middle = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Align BB to 4h timeframe (wait for 4h bar close)
    bb_upper_aligned = align_htf_to_ltf(prices, df_4h, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_4h, bb_lower)
    bb_middle_aligned = align_htf_to_ltf(prices, df_4h, bb_middle)
    
    # === 12h Bollinger Band Width percentile (squeeze detection) ===
    bb_middle_12h = pd.Series(close_12h).rolling(window=20, min_periods=20).mean().values
    bb_std_12h = pd.Series(close_12h).rolling(window=20, min_periods=20).std().values
    bbw_12h = (bb_std_12h * 4) / bb_middle_12h  # BBW = (upper-lower)/middle = 4*std/middle
    # Percentile rank of BBW over 50 periods
    bbw_percentile = pd.Series(bbw_12h).rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    squeeze_condition = bbw_percentile < 20  # BBW in lowest 20% = squeeze
    squeeze_aligned = align_htf_to_ltf(prices, df_12h, squeeze_condition)
    
    # === 4h Volume confirmation (expanding volume) ===
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_expanding = volume_4h > (1.5 * vol_ma_20_4h)
    vol_expanding_aligned = align_htf_to_ltf(prices, df_4h, vol_expanding)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss/time exit
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_in_trade = 0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or
            np.isnan(squeeze_aligned[i]) or
            np.isnan(vol_expanding_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            bars_in_trade = 0
            continue
        
        price = close[i]
        squeeze_ok = squeeze_aligned[i]
        vol_ok = vol_expanding_aligned[i]
        
        # === STOPLOSS LOGIC (ATR-based) ===
        if position != 0:
            atr_4h = np.abs(high_4h - low_4h)
            atr_ma = pd.Series(atr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_4h, atr_ma)
            atr_val = atr_aligned[i]
            if position == 1:  # Long position
                if price < entry_price - 2.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                    bars_in_trade = 0
                    continue
            elif position == -1:  # Short position
                if price > entry_price + 2.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                    bars_in_trade = 0
                    continue
        
        # === TIME-BASED EXIT (max 8 bars) ===
        if position != 0:
            bars_in_trade += 1
            if bars_in_trade >= 8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                bars_in_trade = 0
                continue
        else:
            bars_in_trade = 0
        
        # === EXIT LOGIC (mean reversion to middle band) ===
        if position == 1:  # Long position
            # Exit when price returns to middle band
            if price <= bb_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                bars_in_trade = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price returns to middle band
            if price >= bb_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                bars_in_trade = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require squeeze and volume expansion
            if squeeze_ok and vol_ok:
                # Go long when price breaks above upper band
                if price > bb_upper_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    bars_in_trade = 1
                    continue
                # Go short when price breaks below lower band
                elif price < bb_lower_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    bars_in_trade = 1
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_BB_Squeeze_Breakout_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0