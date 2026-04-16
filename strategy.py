#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA200 trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND price > 1w EMA200 (uptrend) AND volume > 1.5x 20-period 1d average.
# Short when price breaks below Donchian(20) low AND price < 1w EMA200 (downtrend) AND volume > 1.5x 20-period 1d average.
# Exit when price reverses to Donchian(10) opposite level or ATR-based stoploss (2*ATR from entry).
# Uses discrete position size 0.25. Designed to capture strong trending moves with volume confirmation.
# Works in both bull and bear markets by requiring trend alignment (price vs 1w EMA200) and volume confirmation.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: Donchian Channels ===
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # === 1w Indicators: EMA200 Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    price_above_ema200 = close > ema_200_1w_aligned
    price_below_ema200 = close < ema_200_1w_aligned
    
    # === 1d Indicators: ATR for Stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 200 periods needed for EMA200)
    warmup = 200
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or np.isnan(donchian_high_10[i]) or
            np.isnan(donchian_low_10[i]) or np.isnan(volume_spike[i]) or np.isnan(price_above_ema200[i]) or
            np.isnan(price_below_ema200[i]) or np.isnan(atr_1d[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        above_ema200 = price_above_ema200[i]
        below_ema200 = price_below_ema200[i]
        atr_val = atr_1d[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price reverses to Donchian(10) low
            if price <= donchian_low_10[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price reverses to Donchian(10) high
            if price >= donchian_high_10[i]:
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
            # LONG: Price breaks above Donchian(20) high AND price > 1w EMA200 (uptrend) AND volume spike
            if price > donchian_high_20[i] and above_ema200 and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Donchian(20) low AND price < 1w EMA200 (downtrend) AND volume spike
            elif price < donchian_low_20[i] and below_ema200 and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_Donchian20_1wEMA200_VolumeSpike_V1"
timeframe = "1d"
leverage = 1.0